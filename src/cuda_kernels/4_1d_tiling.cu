#include <torch/torch.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void 1d_block_tiling(int M, int N, int K, float alpha,
                                const float *A, const float *B,
                                float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BM * K;                    // row=cRow, col=0
  B += cCol * BN;                        // row=0, col=cCol
  C += cRow * BM * N + cCol * BN; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BK + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BN + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BK;
    B += BK * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      tmp += As[threadRow * BK + dotIdx] *
             Bs[dotIdx * BN + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

torch::Tensor forward_1d_tiling(torch::Tensor A, torch::Tensor B) {
    int M, K, N;

    M = A.size(0);
    K = A.size(1);
    N = B.size(1);

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    torch::Tensor C = torch::zeros({M, N}).to(A.device());

    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim((BM*BN)/TM);
    cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    1d_block_tiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(
        M, N, K,
        1.0, 
        A.data_ptr<float>(), B.data_ptr<float>(), 
        1.0,
        C.data_ptr<float>()
    );

    return C;
}