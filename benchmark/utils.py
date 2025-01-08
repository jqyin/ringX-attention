import re
import os
from collections import defaultdict

batch_seqlen_pattern = re.compile(r'Batch size: (\d+).*?Sequence length: (\d+)', re.DOTALL)
perf_pattern = re.compile(r'# (\w+)_func\s+ngpus: (\d+).*?batch: (\d+), seqlen: (\d+), num_heads: (\d+), head_dim: (\d+).*?\n.*? ([0-9.]+) sec')

def parse_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        
        match = batch_seqlen_pattern.findall(content)
        if not match:
            return []
        
        results = []
        for func, ngpus, batch, seq, num_heads, head_dim, sec in perf_pattern.findall(content):
            results.append((int(batch), int(seq), func, int(ngpus), int(num_heads), int(head_dim), float(sec)))
        
        return results

def process_files(file_list):
    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for file_path in file_list:
        results = parse_file(file_path)
        for batch_size, seqlen, func, ngpus, num_heads, head_dim, sec in results:
            grouped_results[(batch_size, seqlen)][func][ngpus].append((sec, num_heads, head_dim))
    
    for key in grouped_results:
        for func in grouped_results[key]:
            grouped_results[key][func] = dict(sorted(grouped_results[key][func].items()))
    
    return grouped_results

# Calculate FLOPS
def calculate_flops(batch_size, seqlen, ngpus, num_heads, head_dim):
    s = seqlen * ngpus
    h = num_heads * head_dim
    return 8 * batch_size * s * h**2 + 4 * batch_size * s**2 * h

if __name__ == "__main__":
    file_list = [f for f in os.listdir('.') if f.startswith('log.algo')]
    results = process_files(file_list)
    
    for (batch, seqlen), funcs in results.items():
        print(f"Batch size: {batch}, Sequence length: {seqlen}")
        for func, perf_data in funcs.items():
            print(f"  Function: {func}")
            for ngpus, sec_list in perf_data.items():
                for sec, num_heads, head_dim in sec_list:
                    forward_flops = calculate_flops(batch, seqlen, ngpus, num_heads, head_dim)
                    iters = 5
                    TFLOPS = 3*forward_flops/(sec/iters)/1e12 # TFLOPS
                    print(f"    GPUs: {ngpus}, Time: {sec} sec, TFLOPS: {TFLOPS/ngpus:.1f}")

