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
    assert causal == True, "This implementation is intended for causal=True"

    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    global_ranks = [dist.get_global_rank(process_group, i) for i in range(world_size)]

    kv = torch.cat([k, v], dim=0)
    k_size = k.shape[0]
    block_seq_len = q.shape[1] // 2

    kv_buffers = [
        torch.empty_like(kv),
        torch.empty_like(kv)   
    ]
    current_buf_idx = 0

    out, lse = None, None

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

    res_rank = global_ranks[0]

    if rank == res_rank:
        kv_buffers[current_buf_idx][:k_size].copy_(k)
        kv_buffers[current_buf_idx][k_size:].copy_(v)

    broadcast_work = dist.broadcast(
        kv_buffers[current_buf_idx], 
        src=res_rank, 
        group=process_group, 
        async_op=True
    )

    for i in range(world_size):
        broadcast_work.wait()

        kv_buffer = kv_buffers[current_buf_idx]
        k_bcast = kv_buffer[:k_size]
        v_bcast = kv_buffer[k_size:]

        if i == rank:
            block_out, block_lse = flash_forward(q, k_bcast, v_bcast, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif i < rank:
            block_out, block_lse = flash_forward(
                q,
                k_bcast[:, :block_seq_len], 
                v_bcast[:, :block_seq_len],
                causal=False
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = flash_forward(
                q[:, block_seq_len:], 
                k_bcast, 
                v_bcast,
                causal=False
            )
            out, lse = update_out_and_lse(
                out, lse, block_out, block_lse, 
                slice_=(slice(None), slice(block_seq_len, None))
            )

        if i < world_size - 1:
            next_idx = 1 - current_buf_idx
            next_res_rank = global_ranks[i + 1]
            if rank == next_res_rank:
                kv_buffers[next_idx][:k_size].copy_(k)
                kv_buffers[next_idx][k_size:].copy_(v)

            broadcast_work = dist.broadcast(
                kv_buffers[next_idx], 
                src=next_res_rank, 
                group=process_group, 
                async_op=True
            )

            current_buf_idx = next_idx

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
    global_ranks = [dist.get_global_rank(process_group, i) for i in range(world_size)]

    k_size = k.shape[0]  
    kv = torch.cat([k, v], dim=0)  
    block_seq_len = q.shape[1] // 2
    dout_left, dout_right = dout.chunk(2, dim=1)
    q_left, q_right       = q.chunk(2, dim=1)
    out_left, out_right   = out.chunk(2, dim=1)
    lse_left, lse_right   = softmax_lse.chunk(2, dim=2)  

    dq_buffer = torch.empty_like(q)          
    dk_buffer = torch.empty_like(k)         
    dv_buffer = torch.empty_like(v)        
    dq = torch.zeros_like(q, dtype=torch.float32)
    dkv_sum = torch.empty_like(kv, dtype=torch.float32)
    dk, dv = None, None

    kv_buffers = [
        torch.empty_like(kv),  
        torch.empty_like(kv), 
    ]
    current_buffer_idx = 0

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


    res_rank = global_ranks[0]
    if rank == res_rank:
        kv_buffers[current_buffer_idx][:k_size].copy_(k)
        kv_buffers[current_buffer_idx][k_size:].copy_(v)

    broadcast_work = dist.broadcast(
        kv_buffers[current_buffer_idx],
        src=res_rank,
        group=process_group,
        async_op=True
    )

    for i in range(world_size):
        broadcast_work.wait()

        kv_buffer = kv_buffers[current_buffer_idx]
        k_bcast = kv_buffer[:k_size]
        v_bcast = kv_buffer[k_size:]

        if i == rank:
            flash_backward(dout, q, k, v, out, softmax_lse, causal=True)
            dq += dq_buffer
            dkv_sum[:k_size, :].copy_(dk_buffer)
            dkv_sum[k_size:, :].copy_(dv_buffer)

        elif i < rank:
            flash_backward(
                dout, 
                q, 
                k_bcast[:, :block_seq_len],
                v_bcast[:, :block_seq_len],
                out,
                softmax_lse,
                causal=False
            )
            dq += dq_buffer
            dkv_sum[:k_size, :block_seq_len] = dk_buffer[:, :block_seq_len]
            dkv_sum[k_size:, :block_seq_len] = dv_buffer[:, :block_seq_len]

        else: 
            flash_backward(
                dout_right,
                q_right,
                k_bcast,
                v_bcast,
                out_right,
                lse_right,
                causal=False
            )
            dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]
            dkv_sum[:k_size, :].copy_(dk_buffer)
            dkv_sum[k_size:, :].copy_(dv_buffer)

        reduce_work = dist.reduce(
            dkv_sum,
            dst=global_ranks[i],
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True
        )

        if i < world_size - 1:
            next_idx = 1 - current_buffer_idx
            next_res_rank = global_ranks[i + 1]
            if rank == next_res_rank:
                kv_buffers[next_idx][:k_size].copy_(k)
                kv_buffers[next_idx][k_size:].copy_(v)

            broadcast_work = dist.broadcast(
                kv_buffers[next_idx],
                src=next_res_rank,
                group=process_group,
                async_op=True
            )
            current_buffer_idx = next_idx

        if rank == i:
            reduce_work.wait()
            dk = dkv_sum[:k_size].clone()
            dv = dkv_sum[k_size:].clone()

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

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
