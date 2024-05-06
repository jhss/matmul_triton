#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1)/ (N))
/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

torch::Tensor forward_naive(torch::Tensor A, torch::Tensor B) {
    int M, K, N;

    M = A.size(0);
    K = A.size(1);
    N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}).to(A.device());

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    sgemm_naive<<<gridDim, blockDim>>>(
        M, N, K,
        1.0, 
        A.data_ptr<float>(), B.data_ptr<float>(), 
        1.0,
        C.data_ptr<float>()
    );

    return C;
}