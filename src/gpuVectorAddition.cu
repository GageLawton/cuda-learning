#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\\n", cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

__global__ void vectorAdd(float* d_A, float* d_B, float* d_C) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < N) {
        // perform vector addition for element i
        printf("Thread %d is processing element %d\n", threadIdx.x, i);
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main(){
    float h_A[N];
    float h_B[N];
    float h_C[N];

    for( int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));


    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, N * sizeof(float)));

    vectorAdd<<<N/256, 256>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}