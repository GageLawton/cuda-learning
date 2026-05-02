#include <stdio.h>

__global__ void kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    kernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}