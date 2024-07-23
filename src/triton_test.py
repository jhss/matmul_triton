import math
from functools import partial

import torch
import triton
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.utils.cpp_extension import load

from triton_kernels import *
# Step1 (Naive)
#load(name='naive_step1', sources=[''])

def matmul(a, b, matmul_fn, grid_type):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    if grid_type == '1D':
        grid = lambda meta: (M*N,)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 1, 1, 1024
    elif grid_type == '2D':
        grid = lambda meta: (M, N)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32

    matmul_fn[grid](M,N,K, 1.0, 0.0, 
                    a, b, c,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
                    )
    return c

naive_matmul  = partial(matmul, matmul_fn=naive_kernel, grid_type='1D')
global_matmul = partial(matmul, matmul_fn=global_mem_coalsce_kernel, grid_type='1D')
shared_matmul = partial(matmul, matmul_fn=shared_mem_kernel, grid_type='2D')

torch.manual_seed(0)
a = torch.randn((2048, 2048), device='cuda', dtype=torch.float32)
b = torch.randn((2048, 2048), device='cuda', dtype=torch.float32)
output_naive  = matmul(a, b, matmul_fn=naive_kernel, grid_type='1D')
output_global = matmul(a, b, matmul_fn=global_mem_coalsce_kernel, grid_type='1D')
output_shared = matmul(a, b, matmul_fn=shared_mem_kernel, grid_type='2D')
output_torch  = torch.matmul(a, b)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_naive))}\n'
      f'{torch.max(torch.abs(output_torch - output_global))}\n'
      f'{torch.max(torch.abs(output_torch - output_shared))}\n')

print(output_naive[0:10,0])
print(output_global[0:10,0])
print(output_shared[0:10,0])
print(output_torch[0:10,0])

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['square_matrix_size'],
        x_vals=[2**i for i in range(5, 12, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['naive', 'global', 'shared', 'torch'],
        line_names=['Naive', 'Global', 'Shared', 'Torch'],
        styles=[('blue', '-'), ('orange','-'), ('black', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='matmul-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},
    ))

def benchmark(square_matrix_size, provider):
    sz = square_matrix_size
    a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'naive': ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a,b), quantiles=quantiles)
    if provider == 'global': ms, min_ms, max_ms = triton.testing.do_bench(lambda: global_matmul(a,b), quantiles=quantiles)
    if provider == 'shared': ms, min_ms, max_ms = triton.testing.do_bench(lambda: shared_matmul(a,b), quantiles=quantiles)
    if provider == 'torch': ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a,b), quantiles=quantiles)

    gbps = lambda ms: 12* sz / ms * 1e-6
    return ms, gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
    