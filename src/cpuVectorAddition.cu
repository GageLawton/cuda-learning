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

void cpuVectorAdd(float* A, float* B, float* C, int n) {
    // your addition logic here
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float A[N];
    float B[N];
    float C[N];

    for( int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    cpuVectorAdd(A, B, C, N);

    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    return 0;
}