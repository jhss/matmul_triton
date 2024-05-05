import torch

import triton
import triton.language as tl

@triton.jit
def naive_kernel(M, N, K, alpha, beta,
                 a_ptr, b_ptr, c_ptr,
                 stride_a_row, stride_a_col,
                 stride_b_row, stride_b_col,
                 stride_c_row, stride_c_col,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):

    pid = tl.program_id(axis=0)
    row_idx = pid % M
    col_idx = pid // M

    offset_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + row_idx * stride_a_row + offset_k * stride_a_col
    b_ptrs = b_ptr + col_idx * stride_b_col + offset_k * stride_b_row

    accumulator = 0.0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        A = tl.load(a_ptrs, mask=offset_k < K - k * BLOCK_SIZE_K)
        B = tl.load(b_ptrs, mask=offset_k < K - k * BLOCK_SIZE_K)
        accumulator += tl.sum(A*B)
        a_ptrs += BLOCK_SIZE_K * stride_a_col
        b_ptrs += BLOCK_SIZE_K * stride_b_row

    c_ptrs = c_ptr + row_idx * stride_c_row + col_idx * stride_c_col
    tl.store(c_ptrs, accumulator)