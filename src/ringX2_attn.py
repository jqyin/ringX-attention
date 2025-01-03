import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from utils import get_default_args

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

    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    out, lse, lse_max = None, None, None
    q_buffer = torch.empty_like(q).contiguous()
    def flash_forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "window_size": window_size,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        out, _, _, _, _, lse, _, _ = _flash_attn_forward(**params)
        return out, lse

    for i in range(world_size - 1, -1, -1):
        q_buffer[:] = q
        res_rank = dist.get_global_rank(process_group, i)
        dist.broadcast(q_buffer, src=res_rank, group=process_group) 
        
        if not causal or rank <= i:
            loc_out, loc_lse = flash_forward(q_buffer, k, v, causal=causal and rank == i)   
            loc_out = loc_out.to(torch.float32)
            loc_lse = loc_lse.transpose(-2, -1).unsqueeze(dim=-1).contiguous()
            if lse_max is None: 
                lse_max = loc_lse.clone()
            else:
                lse_max[:] = loc_lse
        else:
            lse_max[:] = -torch.finfo(q.dtype).max
        dist.all_reduce(lse_max, op=dist.ReduceOp.MAX, group=process_group)
        if not causal or rank <= i:
            den = torch.exp(loc_lse - lse_max)
            num = loc_out * den 
        else:
            den.zero_()
            num.zero_()
        dist.reduce(num, dst=res_rank, op=dist.ReduceOp.SUM, group=process_group)
        dist.reduce(den, dst=res_rank, op=dist.ReduceOp.SUM, group=process_group)
        if rank == i: 
            out = num.div_(den.clamp(min=1e-8)).to(q.dtype)
            lse = (torch.log(den) + lse_max).squeeze(dim=-1).transpose(1, 2).contiguous()
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
    assert causal == True, "current implementation is for causal only"
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    dq, dk, dv = None, None, None
    dq_buffer = torch.empty_like(q)
    dk_buffer = torch.empty_like(k)
    dv_buffer = torch.empty_like(v)
    kv = torch.cat([k,v], dim=0)
    kv_buffer = torch.empty_like(kv)
    k_size0 = k.shape[0]
    dkv_sum = torch.empty_like(kv, dtype=torch.float32).contiguous()
    dq_sum = torch.empty_like(q, dtype=torch.float32).contiguous()

    def flash_backward(dout, q, k, v, out, softmax_lse, causal):
        params = get_default_args(_flash_attn_backward).copy()
        params.update(
            {
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "softmax_lse": softmax_lse,
                "dq": dq_buffer,
                "dk": dk_buffer,
                "dv": dv_buffer,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "window_size": window_size,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
            }
        )
        _flash_attn_backward(**params)

    for i in range(world_size):
        kv_buffer[:k_size0].copy_(k)
        kv_buffer[k_size0:].copy_(v)
        res_rank = dist.get_global_rank(process_group, i)
        dist.broadcast(kv_buffer, src=res_rank, group=process_group)
       
        if rank == i:
            flash_backward(dout, q, k, v, out, softmax_lse, causal=True)
        elif rank > i:
            flash_backward(dout, q, kv_buffer[:k_size0], kv_buffer[k_size0:], out, softmax_lse, causal=False)
            if dq is None:
                dq = dq_buffer.to(torch.float32)
            else:
                dq += dq_buffer
            dq_buffer.zero_()
        else:
            dq_buffer.zero_()
            dk_buffer.zero_()
            dv_buffer.zero_()

        dkv_sum[:k_size0].copy_(dk_buffer)
        dkv_sum[k_size0:].copy_(dv_buffer)
        dq_sum.copy_(dq_buffer)
        dist.reduce(dkv_sum, dst=res_rank, op=dist.ReduceOp.SUM, group=process_group)
        dist.reduce(dq_sum, dst=res_rank, op=dist.ReduceOp.SUM, group=process_group)
        if rank == i: 
            if dq is None:
                dq = dq_sum.clone()
            else:
                dq += dq_sum
            dk = dkv_sum[:k_size0].clone()
            dv = dkv_sum[k_size0:].clone()

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
