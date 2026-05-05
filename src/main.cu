#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
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
        d_C[i] = d_A[i] + d_B[i];
    }
}

void cpuVectorAdd(float* A, float* B, float* C, int n) {
    // your addition logic here
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(){
    // Measure CPU time
    auto start = std::chrono::high_resolution_clock::now();

    float h_A[N];
    float h_B[N];
    float h_C[N];

    for( int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    cpuVectorAdd(h_A, h_B, h_C, N);

    auto end = std::chrono::high_resolution_clock::now();

    // Measure GPU time

    // Warmup - force CUDA context initialization
    float* warmup;
    cudaMalloc(&warmup, sizeof(float));
    cudaFree(warmup);

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

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

    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print results
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }   
    auto cpuTime = std::chrono::duration<double, std::micro>(end - start).count();
    printf("CPU Time: %.3f ms\n", cpuTime / 1000.0);
    printf("GPU Time: %.3f ms\n", gpuTime);
    
    return 0;
}