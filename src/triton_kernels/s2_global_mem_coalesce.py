import torch

import triton
import triton.language as tl

@triton.jit
def global_mem_coalsce_kernel(M, N, K: tl.constexpr, alpha, beta,
                              a_ptr, b_ptr, c_ptr):
    y = tl.program_id(axis=0)
    x = tl.program_id(axis=1)

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