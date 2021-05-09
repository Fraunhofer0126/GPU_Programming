#include<stdio.h>
__global__ void hello_from_gpu()
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}
int main(void)
{
    hello_from_gpu<<<2, 3>>>();
    cudaDeviceReset();
    return 0;
}