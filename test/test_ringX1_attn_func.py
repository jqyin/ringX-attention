from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
from ringX1_attn import ringX_attn_func
from utils import log, set_seed


def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.contiguous()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3824
    num_heads = 5
    head_dim = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert causal
    assert seqlen % (2 * world_size) == 0
    assert head_dim  % 8 == 0
    q = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    dout = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_q = extract_local(q, rank, world_size).detach().clone()
    local_k = extract_local(k, rank, world_size).detach().clone()
    local_v = extract_local(v, rank, world_size).detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    local_dout = extract_local(dout, rank, world_size).detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_func(
        q,k,v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = extract_local(out, rank, world_size)
    local_lse = extract_local(lse, rank, world_size, dim=2)

    ring_out, ring_lse, _ = ringX_attn_func(
        local_q, local_k, local_v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad

    local_dq = extract_local(dq, rank, world_size)
    local_dk = extract_local(dk, rank, world_size)
    local_dv = extract_local(dv, rank, world_size)

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad

    #log("local_dqkv", local_dqkv)
    log("dq diff", local_dq[:] - ring_dq[:, 0])
    log("dk diff", local_dk[:] - ring_dk[:, 1])
    log("dv diff", local_dv[:] - ring_dv[:, 2])
