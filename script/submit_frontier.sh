#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 0:20:00
#SBATCH -A stf218
#SBATCH -N 64
#SBATCH -q debug 
#SBATCH --exclusive
##SBATCH --ntasks-per-node=8
#SBATCH -o tft.o%j
#SBATCH -e tft.e%j

#algo="ringX2_attn"   
algo="${@}"   

module use /sw/aaims/crusher/modulefiles
module load xforge
export PYTHONPATH=../src
export NCCL_SOCKET_IFNAME=hsn
export MIOPEN_DISABLE_CACHE=1
export OMP_NUM_THREADS=7

export FI_PROVIDER=cxi
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RDZV_PROTO=alt_read
export NCCL_CROSS_NIC=1
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_DISABLE_CQ_HUGETLB=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=disabled
export NCCL_ALGO=Tree

batch_sizes=(32 64 128 256)
seq_lengths=(8192 4096 2048 1024)

num_heads=32
head_dim=128
causal=false

for ((i=0; i<${#batch_sizes[@]}; i++)); do
  batch_size=${batch_sizes[i]}
  seq_length=${seq_lengths[i]}

  CMD="python ../benchmark/benchmark_ringX_func.py \
    --batch_size $batch_size \
    --seq_length $seq_length \
    --num_heads $num_heads \
    --head_dim $head_dim \
    --module $algo"
  if [ "$causal" = true ]; then
    CMD="$CMD --causal"
  fi

  srun --nodes=${SLURM_NNODES} \
     --network=disable_rdzv_get \
     --ntasks=$((SLURM_NNODES*8)) \
     --ntasks-per-gpu=1 --gpus-per-node=8 --gpu-bind=closest -c7 \
     bash -c "source setup_dist_vars.sh; $CMD" >> log.algo-${algo}_n${SLURM_NNODES}_${SLURM_JOB_ID}
done
