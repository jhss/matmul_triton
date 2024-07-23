#include <torch/torch.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void block_tiling_1d(int M, int N, int K, float alpha,
                                const float *A, const float *B,
                                float beta, float *C) {
  // Qeustion: (flip x and y) ??
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BN;
  const uint threadRow = threadIdx.x / BN;

  // advance pointers to the starting positions
  A += cRow * BM * K;                    // row=cRow, col=0
  B += cCol * BN;                        // row=0, col=cCol
  C += cRow * BM * N + cCol * BN; // row=cRow, col=cCol

  const uint innerRowA = threadIdx.x % BK;
  const uint innerColA = threadIdx.x / BK;
  const uint innerRowB = threadIdx.x % BN;
  const uint innerColB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BK;
    B += BK * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; resIdx++) {
        threadResults[resIdx] += As[(threadRow*TM+resIdx)*BK + dotIdx] * Btmp;
      }
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
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

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM*BN)/TM);
    block_tiling_1d<BM, BN, BK, TM><<<gridDim, blockDim>>>(
        M, N, K,
        1.0, 
        A.data_ptr<float>(), B.data_ptr<float>(), 
        1.0,
        C.data_ptr<float>()
    );

    return C;
}