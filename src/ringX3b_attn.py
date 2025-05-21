import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from utils import get_default_args, update_out_and_lse
try:
    from pccl import _all_gather, all_gather_2D, reduce_scatter_2D
    HAS_PCCL = True
    num_gpus_per_node = 8 
except ImportError:
    HAS_PCCL = False

def ringX_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == False, "ringX3b is intended for causal=False"

    if HAS_PCCL:    
        inner_rank, outer_rank = process_group.get_rank() 
        inner_size, outer_size = process_group.get_world_size() 
        rank = outer_rank *num_gpus_per_node + inner_rank
        world_size = inner_size*outer_size
        assert world_size%8 == 0 and world_size >= 8, "ringX is intended for multi-node CP"
    else:
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)

    kv = torch.cat([k, v], dim=0)
    if HAS_PCCL:
        kv_shape = kv.shape
        kv_flat = kv.contiguous().view(-1)
        numel_per_rank = kv_flat.numel()
        _buffer = torch.empty(numel_per_rank * world_size//num_gpus_per_node,
                          dtype=kv_flat.dtype,
                          device=kv_flat.device)
        gather_kv_step1 = dist.all_gather_into_tensor(_buffer, kv_flat, group=process_group.get_outer_group(), async_op=True) 
         
        kv_all_flat = torch.empty(world_size * numel_per_rank,
                          dtype=kv_flat.dtype,
                          device=kv_flat.device)
        kv = kv_flat.view(kv.shape)
    else:
        kv_all = [torch.empty_like(kv) for _ in range(world_size)]
        gather_kv = dist.all_gather(kv_all, kv, group=process_group, async_op=True)

    k_size = k.shape[0]

    out = None
    lse = None
    def flash_forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                     {
                         "window_size_left": window_size[0],
                         "window_size_right": window_size[1],
                     }
            )
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        outputs = _flash_attn_forward(**params)
        if len(outputs) == 8:
            out, _, _, _, _, lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            out, lse, _, _ = outputs

        return out, lse

    block_out, block_lse = flash_forward(q, k, v, causal=False)
    out, lse = update_out_and_lse(out, lse, block_out, block_lse)

    if HAS_PCCL:    
        gather_kv_step1.wait()
        dist.all_gather_into_tensor(kv_all_flat, _buffer, group=process_group.get_inner_group(), async_op=False) 
        output_unpermuted = kv_all_flat.view(num_gpus_per_node, world_size//num_gpus_per_node, -1).transpose(0, 1).reshape(-1)
        kv_all_flat.copy_(output_unpermuted)
        kv_all = [
            kv_all_flat[i * numel_per_rank : (i + 1) * numel_per_rank].view(kv_shape)
            for i in range(world_size)
        ]
    else:
        gather_kv.wait()


    for i in range(world_size):
        if i == rank: 
            continue 
        block_out, block_lse = flash_forward(q, kv_all[i][:k_size,:], kv_all[i][k_size:,:], causal=False)
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ringX_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
): 
    
    if HAS_PCCL:    
        inner_rank, outer_rank = process_group.get_rank()
        inner_size, outer_size = process_group.get_world_size()
        rank = outer_rank *num_gpus_per_node + inner_rank
        world_size = inner_size*outer_size
        assert world_size%8 == 0 and world_size >= 8, "ringX is intended for multi-node CP"
    else:
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)

    kv = torch.cat([k, v], dim=0)
    if HAS_PCCL:
        kv_shape = kv.shape
        kv_flat = kv.contiguous().view(-1)
        numel_per_rank = kv_flat.numel()
        _buffer = torch.empty(numel_per_rank * world_size//num_gpus_per_node,
                          dtype=kv_flat.dtype,
                          device=kv_flat.device)
        gather_kv_step1 = dist.all_gather_into_tensor(_buffer, kv_flat, group=process_group.get_outer_group(), async_op=True) 

        kv_all_flat = torch.empty(world_size * numel_per_rank,
                          dtype=kv_flat.dtype,
                          device=kv_flat.device)

        kv = kv_flat.view(kv.shape)
    else:
        kv_all = [torch.empty_like(kv) for _ in range(world_size)]
        gather_kv = dist.all_gather(kv_all, kv, group=process_group, async_op=True)

    k_size = k.shape[0]
    dq_buffer = torch.empty_like(q)
    dk_buffer = torch.empty_like(k)
    dv_buffer = torch.empty_like(v)
    dk_flat = dk_buffer.view(-1)
    dv_flat = dv_buffer.view(-1)
    dkv = torch.empty(dk_flat.numel() + dv_flat.numel(), dtype=torch.float32, device=q.device)
    chunk_size = dkv.numel()
    dkv_all = torch.zeros(world_size * chunk_size, dtype=torch.float32, device=q.device)
    dk_size = dk_flat.numel()
 
    def flash_backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        params = get_default_args(_flash_attn_backward).copy()
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                     {
                         "window_size_left": window_size[0],
                         "window_size_right": window_size[1],
                     }
            )
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
        params.update(
            {
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "softmax_lse": softmax_lse,
                "dq": dq_buffer[:, :seqlen_q],
                "dk": dk_buffer[:, :seqlen_kv],
                "dv": dv_buffer[:, :seqlen_kv],
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
                "rng_state": rng_state,
            }
        )
        _flash_attn_backward(**params)


    flash_backward(dout, q, k, v, out, softmax_lse, causal=False)
    dq = dq_buffer.to(torch.float32)
    offset = rank * chunk_size
    dkv_all[offset : offset + dk_size].copy_(dk_flat)
    dkv_all[offset + dk_size : offset + chunk_size].copy_(dv_flat)

    if HAS_PCCL:    
        gather_kv_step1.wait()
        dist.all_gather_into_tensor(kv_all_flat, _buffer, group=process_group.get_inner_group(), async_op=False)
        output_unpermuted = kv_all_flat.view(num_gpus_per_node, world_size//num_gpus_per_node, -1).transpose(0, 1).reshape(-1)
        kv_all_flat.copy_(output_unpermuted)
        kv_all = [
            kv_all_flat[i * numel_per_rank : (i + 1) * numel_per_rank].view(kv_shape)
            for i in range(world_size)
        ]
    else:
        gather_kv.wait()

    for i in range(world_size):
        if i == rank:
            continue
        flash_backward(dout, q, kv_all[i][:k_size], kv_all[i][k_size:], out, softmax_lse, causal=False)
        dq += dq_buffer
        offset = i * chunk_size
        dkv_all[offset : offset + dk_size].copy_(dk_flat)
        dkv_all[offset + dk_size : offset + chunk_size].copy_(dv_flat)

    if HAS_PCCL:
        reduce_scatter_2D(dkv, dkv_all, group=process_group, use_rh=True, use_pccl_cpp_backend=True)
    else:
        dist.reduce_scatter_tensor(dkv, dkv_all, op=dist.ReduceOp.SUM, group=process_group)

    return dq.to(q.dtype), dkv[:dk_size].view_as(k).to(k.dtype), dkv[dk_size:].view_as(v).to(v.dtype)


class RingXAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ringX_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ringX_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def ringX_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingXAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
