import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from utils import get_default_args, update_out_and_lse


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
    assert causal == True, "ringX1 is intended for causal=False"

    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    kv = torch.cat([k, v], dim=0)
    kv_all = [torch.empty_like(kv) for _ in range(world_size)]
    gather_kv = dist.all_gather(kv_all, kv, group=process_group, async_op=True)

    block_seq_len = q.shape[1] // 2
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

    block_out, block_lse = flash_forward(q, k, v, causal=True)
    out, lse = update_out_and_lse(out, lse, block_out, block_lse)
    gather_kv.wait() 

    for i in range(world_size):
        if i == rank: 
            continue 
        if i < rank:
            block_out, block_lse = flash_forward(q, kv_all[i][:k_size,:block_seq_len], kv_all[i][k_size:,:block_seq_len], causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:  
            block_out, block_lse = flash_forward(q[:,block_seq_len:], kv_all[i][:k_size], kv_all[i][k_size:], causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse, slice_=(slice(None), slice(block_seq_len, None)))
    
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
    
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    kv = torch.cat([k, v], dim=0)
    kv_all = [torch.empty_like(kv) for _ in range(world_size)]
    gather_kv = dist.all_gather(kv_all, kv, group=process_group, async_op=True)

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2
    k_size = k.shape[0]
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    dkv = torch.cat([dk_buffer, dv_buffer], dim=0)
    dkv_all = [torch.zeros_like(dkv) for _ in range(world_size)]
    dk_size = dk_buffer.shape[0]
 
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


    flash_backward(dout, q, k, v, out, softmax_lse, causal=True)
    dq = dq_buffer.to(torch.float32)
    dkv_all[rank][:dk_size] = dk_buffer
    dkv_all[rank][dk_size:] = dv_buffer

    gather_kv.wait() 

    for i in range(world_size):
        if i == rank:
            continue 
        if i < rank:
            flash_backward(dout, q, kv_all[i][:k_size,:block_seq_len], kv_all[i][k_size:,:block_seq_len], out, softmax_lse, causal=False)
            dq += dq_buffer
            dkv_all[i][:dk_size, :block_seq_len] = dk_buffer[:, :block_seq_len]
            dkv_all[i][dk_size:, :block_seq_len] = dv_buffer[:, :block_seq_len]
        else:
            flash_backward(dout1, q1, kv_all[i][:k_size], kv_all[i][k_size:], out1, softmax_lse1, causal=False)
            dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]
            dkv_all[i][:dk_size] = dk_buffer
            dkv_all[i][dk_size:] = dv_buffer

    dist.reduce_scatter(dkv, dkv_all, op=dist.ReduceOp.SUM, group=process_group)

    return dq.to(q.dtype), dkv[:dk_size].to(k.dtype), dkv[dk_size:].to(v.dtype)

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
