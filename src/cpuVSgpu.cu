#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

__global__ void vectorAdd(float* d_A, float* d_B, float* d_C, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

void cpuVectorAdd(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(){
    // Warmup - do this ONCE before the loop
    float* warmup;
    cudaMalloc(&warmup, sizeof(float));
    cudaFree(warmup);

    int sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-12s %-12s %-12s\n", "N", "CPU (ms)", "GPU (ms)");
    printf("----------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];

        // Allocate host memory
        float* h_A = new float[n];
        float* h_B = new float[n];
        float* h_C = new float[n];

        // Initialize
        for (int i = 0; i < n; i++) {
            h_A[i] = i * 1.0f;
            h_B[i] = i * 2.0f;
        }

        // CPU benchmark
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuVectorAdd(h_A, h_B, h_C, n);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        // GPU benchmark
        float* d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_C, n * sizeof(float)));

        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
        CUDA_CHECK(cudaEventRecord(startEvent, 0));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, n * sizeof(float)));

        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stopEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float gpuTime = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, startEvent, stopEvent));
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));

        printf("%-12d %-12.3f %-12.3f\n", n, cpuTime, gpuTime);

        // Free everything inside the loop
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }

    return 0;
}