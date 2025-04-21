## Scaling Ring Attention on HPC 

This repo supports optimized implementations of [ring flash attention](https://github.com/zhuzilin/ring-flash-attention) on HPC:

- v1: replace the send-recv (k, v) with all-gather (k, v) collective
- v2: v1 with all-gather kv
- v3: tree decoding
- v4: tree forward and backward
- v5: async tree
- v6: opt tree backward 
- v7: v6 with all-gather kv

## Communication Pattern and Memory Footprint

| Algo  | Comm.                                                                                              | Memory |
|-------|----------------------------------------------------------------------------------------------------|--------|
| v1    | all-gather(k, v)                                                                                   | O(S)   |
| v2    | all-gather(kv)                                                                                     | O(S)   |
| v3    | Forward: broadcast(q) all-reduce(lse) reduce(lse, out)   Backward: broadcast(k,v) reduce(dq,dk,dv) | O(S/N) |
| v4    |                                                                                                    |        |
| v5    |                                                                                                    |        |
| v6    |                                                                                                    |        |
| v7    | Forward: broadcast(q) all-reduce(lse) reduce(lse,out)  Backward: all-gather(k,v) reduce(dkv)       | O(S)   |

