import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from ringX1_attn import ringX1_attn_func as ringX_attn_func
from utils import log, set_seed


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3816
    num_heads = 5
    head_dim = 128
    dropout_p = 0
    causal = False
    deterministic = False

    assert not causal
    assert seqlen % world_size == 0
    assert head_dim % 8 == 0

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

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

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

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    ring_out, ring_lse, _ = ringX_attn_func(
        local_q, local_k, local_v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
        group=dist.group.WORLD,
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
    local_dq = dq.chunk(world_size, dim=1)[rank]
    local_dk = dk.chunk(world_size, dim=1)[rank]
    local_dv = dv.chunk(world_size, dim=1)[rank]

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad

    log("dq diff", local_dq[:] - ring_dq[:])
    log("dk diff", local_dk[:] - ring_dk[:])
    log("dv diff", local_dv[:] - ring_dv[:])

