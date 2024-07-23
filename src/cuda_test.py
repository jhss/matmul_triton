import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

kernel_path = "./cuda_kernels"
# Load the CUDA kernel as a python module
cuda_matmul_kernel = load(name='cuda_matmul', 
                          sources=[f'{kernel_path}/main.cpp', 
                                   f'{kernel_path}/1_naive.cu', 
                                   f'{kernel_path}/2_global_mem_coalesce.cu',
                                   f'{kernel_path}/3_shared.cu',
                                   f'{kernel_path}/4_1d_tiling.cu'], 
                          extra_cuda_cflags=['-O2'])

torch.manual_seed(0)
M, K, N = 2048, 2048, 2048
a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)

# print("before: ", a[0:10, 0])
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     result = cuda_matmul_kernel.forward_naive(a,b)
# # print("after: ", a[0:10, 0])
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
result_torch = torch.matmul(a,b)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    result2 = cuda_matmul_kernel.forward_global_coalesce(a,b)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    result3 = cuda_matmul_kernel.forward_shared(a,b.permute([1,0]))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    result4 = cuda_matmul_kernel.forward_1d_tiling(a,b.permute([1,0]))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(result[0:10, 0])
print(result2[0:10, 0])
print(result3[0:10, 0])
print(result4[0:10, 0])
result_torch = torch.matmul(a,b)
print("all close 1: ", torch.allclose(result_torch, result, rtol=5e-2, atol=5e-2))
print("all close 2: ", torch.allclose(result_torch, result2, rtol=5e-2, atol=5e-2))
print("all close 3: ", torch.allclose(result_torch, result3, rtol=5e-2, atol=5e-2))
print("all close 4: ", torch.allclose(result_torch, result4, rtol=5e-2, atol=5e-2))