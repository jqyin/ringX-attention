from flash_attn import flash_attn_func
import os
import torch
import torch.distributed as dist
from ring_flash_attn import (
     ring_flash_attn_func,
#    ring_flash_attn_kvpacked_func,
#    zigzag_ring_flash_attn_kvpacked_func,
#    stripe_flash_attn_kvpacked_func,
)
#from ringX2_attn import ringX_attn_func
import argparse, importlib

def benchmark(args, f, warmup_iter=1, num_iter=100, forward_only=True, log=True, profile=False):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    batch_size = args.batch_size
    deterministic = False
    # config of llama3 8B
    seqlen = args.seq_length
    num_heads = args.num_heads
    head_dim = args.head_dim
    causal = args.causal

    assert seqlen % (2 * world_size) == 0
    assert head_dim % 8 == 0

    if rank == 0: 
        print(f"ngpus: {world_size}, causal: {causal}, batch: {batch_size}, seqlen: {seqlen}, num_heads: {num_heads}, head_dim: {head_dim}")
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
 
    dout = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{dist.get_rank()}"
                )
            ),
        )


    for _ in range(warmup_iter):
        q.grad = None
        k.grad = None
        v.grad = None
        out = f(
            q,
            k,
            v,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=False,
            group=dist.group.WORLD,
        )
        out.backward(dout)


    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(
                    q,
                    k,
                    v,
                    causal=causal,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                    group=dist.group.WORLD,
                )
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            out = f(
                q,
                k,
                v,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
                group=dist.group.WORLD,
            )
            out.backward(dout)
            if profile:
                profiler.step()

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if profile:
        profiler.stop()

    if rank == 0 and log:
        print(f"{num_iter / time:.6f} iter/s, {time:.3f} sec")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training or inference.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--module", type=str, required=True, help="Module name to import the function.")
    parser.add_argument("--causal", action='store_true', help="Enable causal attention masking.")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    args = parser.parse_args()
    try:
        module = importlib.import_module(args.module)
        ringX_attn_func = getattr(module, "ringX_attn_func")
    except ModuleNotFoundError:
        print(f"Error: Module '{args.module}' not found.") 

    if rank == 0:
        print(f"Algo: {args.module}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Number of heads: {args.num_heads}")
        print(f"Head dimension: {args.head_dim}")

    forward_only = False
    profile = False
    num_iter = 5

    for f in [
        #flash_attn_kvpacked_func,
        #ring_flash_attn_kvpacked_func,
        #zigzag_ring_flash_attn_kvpacked_func,
        #stripe_flash_attn_kvpacked_func,
        
        ringX_attn_func,
        ring_flash_attn_func,
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        benchmark(
            args, f, forward_only=forward_only, num_iter=num_iter, log=True, profile=profile
        )
