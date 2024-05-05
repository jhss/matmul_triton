import torch
import triton
import triton.language as tl

@triton.jit
def shared_mem_kernel(M, N, K, alpha, beta,
                      a_ptr, b_ptr, c_ptr,
                      stride_a_row, stride_a_col,
                      stride_b_row, stride_b_col,
                      stride_c_row, stride_c_col,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    
    c_row = tl.program_id(axis=0)
    c_col = tl.program_id(axis=1)

    offset_a = (c_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # Boundary Check
    offset_b = (c_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_c = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offset_a[:, None] * stride_a_row + offset_c[None,:] * stride_a_col)
    b_ptrs = b_ptr + (offset_c[:, None] * stride_b_row + offset_b[None,:] * stride_b_col)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        As = tl.load(a_ptrs, mask=offset_c[None, :] < K - k_idx * BLOCK_SIZE_K, other=0.0)
        Bs = tl.load(b_ptrs, mask=offset_c[:, None] < K - k_idx * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(As, Bs, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_a_col
        b_ptrs += BLOCK_SIZE_K * stride_b_row

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + (offset_a[:,None] * stride_c_row + offset_b[None,:] * stride_c_col)
    c_mask = (offset_a[:,None] < M) & (offset_b[None,:] < N)
    tl.store(c_ptrs, c, mask=c_mask)

