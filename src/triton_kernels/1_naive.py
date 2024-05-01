import torch

import triton
import triton.language as tl

@triton.jit
def naive_kernel(M, N, K: tl.constexpr, alpha, beta, 
                 a_ptr, b_ptr, c_ptr):
    x = tl.program_id(axis=0)
    y = tl.program_id(axis=1)

    offset_a = x * K + tl.arange(0, K)
    offset_b = y + tl.arange(0, K) * N
    a_ptrs = a_ptr + offset_a
    b_ptrs = b_ptr + offset_b

    A = tl.load(a_ptrs)
    B = tl.load(b_ptrs)

    mul_val = A * B
    accumulator = tl.sum(mul_val, axis=0)

    offset_c = x*N + y;
    c_ptrs = c_ptr + offset_c
    C = tl.load(c_ptrs)
    tl.store(c_ptrs, alpha * accumulator + beta * C)

def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (M, N)
    naive_kernel[grid](M,N,K, 1.0, 0.0, a, b, c)
    return c

torch.manual_seed(0)
a = torch.randn((512, 256), device='cuda', dtype=torch.float16)
b = torch.randn((256, 512), device='cuda', dtype=torch.float16)
output_triton = matmul(a, b)
output_torch  = torch.matmul(a, b)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

print(output_triton[0:10,0])
print(output_torch[0:10,0])
