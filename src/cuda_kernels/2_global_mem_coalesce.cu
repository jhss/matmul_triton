#include <torch/torch.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1)/ (N))

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  //const int cCol = blockIdx.y * BLOCKSIZE + blockDim.y * blockIdx.y + (threadIdx.x % BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

torch::Tensor forward_global_coalesce(torch::Tensor A, torch::Tensor B) {
    int M, K, N;

    M = A.size(0);
    K = A.size(1);
    N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}).to(A.device());

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 *32);
    sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(
        M, N, K,
        1.0, 
        A.data_ptr<float>(), B.data_ptr<float>(), 
        1.0,
        C.data_ptr<float>()
    );

    return C;
}