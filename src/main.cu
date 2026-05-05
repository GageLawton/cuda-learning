#include <stdio.h>
#include <cuda_runtime.h>

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

__global__ void initArray(float* arr, float value, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        arr[i] = value * i;
    }
}

int main(){
    // Warmup
    float* warmup;
    cudaMalloc(&warmup, sizeof(float));
    cudaFree(warmup);

    int sizes[] = {
        1024, 2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288, 1048576,
        4194304, 16777216, 67108864
    };
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    float h2dTime = 0.0f;
    float kernelTime = 0.0f;
    float d2hTime = 0.0f;

    printf("%-12s %-12s %-12s %-12s %-12s\n", "N", "H2D (ms)", "Kernel (ms)", "D2H (ms)", "Total (ms)");
    printf("------------------------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];

        // Allocate host memory
        float* h_C = new float[n];

        // Allocate device memory
        float* d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_C, n * sizeof(float)));

        // Initialize arrays directly on GPU - no H2D needed for input data
        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        initArray<<<blocks, threadsPerBlock>>>(d_A, 1.0f, n);
        initArray<<<blocks, threadsPerBlock>>>(d_B, 2.0f, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Event one - H2D transfer time
        // Note: since we init on GPU, we measure a dummy H2D to still capture PCIe cost
        float* h_dummy = new float[n];
        cudaEvent_t h2dStart, h2dStop;
        CUDA_CHECK(cudaEventCreate(&h2dStart));
        CUDA_CHECK(cudaEventCreate(&h2dStop));
        CUDA_CHECK(cudaEventRecord(h2dStart, 0));
        CUDA_CHECK(cudaMemcpy(d_A, h_dummy, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_dummy, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(h2dStop));
        CUDA_CHECK(cudaEventSynchronize(h2dStop));
        CUDA_CHECK(cudaEventElapsedTime(&h2dTime, h2dStart, h2dStop));
        CUDA_CHECK(cudaEventDestroy(h2dStart));
        CUDA_CHECK(cudaEventDestroy(h2dStop));
        delete[] h_dummy;

        // Event two - kernel execution time
        cudaEvent_t kernelStart, kernelStop;
        CUDA_CHECK(cudaEventCreate(&kernelStart));
        CUDA_CHECK(cudaEventCreate(&kernelStop));
        CUDA_CHECK(cudaEventRecord(kernelStart, 0));
        vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));
        CUDA_CHECK(cudaEventDestroy(kernelStart));
        CUDA_CHECK(cudaEventDestroy(kernelStop));

        // Event three - D2H transfer time
        cudaEvent_t d2hStart, d2hStop;
        CUDA_CHECK(cudaEventCreate(&d2hStart));
        CUDA_CHECK(cudaEventCreate(&d2hStop));
        CUDA_CHECK(cudaEventRecord(d2hStart, 0));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(d2hStop));
        CUDA_CHECK(cudaEventSynchronize(d2hStop));
        CUDA_CHECK(cudaEventElapsedTime(&d2hTime, d2hStart, d2hStop));
        CUDA_CHECK(cudaEventDestroy(d2hStart));
        CUDA_CHECK(cudaEventDestroy(d2hStop));

        float totalTime = h2dTime + kernelTime + d2hTime;
        printf("%-12d %-12.3f %-12.3f %-12.3f %-12.3f\n", n, h2dTime, kernelTime, d2hTime, totalTime);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_C;
    }

    return 0;
}