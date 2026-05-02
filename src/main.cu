#include <stdio.h>

__global__ void test() {
    printf("GPU thread %d\n", threadIdx.x);
}

int main() {
    test<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}