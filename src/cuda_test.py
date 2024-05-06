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
                                   f'{kernel_path}/3_shared.cu'], 
                          extra_cuda_cflags=['-O2'])

torch.manual_seed(0)
a = torch.randn((2048, 2048), device='cuda', dtype=torch.float32)
b = torch.randn((2048, 2048), device='cuda', dtype=torch.float32)

# print("before: ", a[0:10, 0])
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    result = cuda_matmul_kernel.forward_naive(a,b)
# print("after: ", a[0:10, 0])
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    result2 = cuda_matmul_kernel.forward_global_coalesce(a,b)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    result3 = cuda_matmul_kernel.forward_shared(a,b)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
#print(result[0:10, 0])
#print(result2[0:10, 0])
print(result3[0:10, 0])
result_torch = torch.matmul(a,b)
#print("all close: ", torch.allclose(result, result2, rtol=5e-2, atol=5e-2))