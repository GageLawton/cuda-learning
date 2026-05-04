#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\\n", cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

__global__ void test() {
    printf("GPU thread %d\n", threadIdx.x);
}

int main() {
    test<<<1, 8>>>();

    CUDA_CHECK(cudaGetLastError());        // check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());   // check runtime errors

    return 0;
}