#include<stdio.h>
__global__ void hello_from_gpu()
{
    int gDim = gridDim.x;
    int bDim = blockDim.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    printf("Hello World from block %d/%d and thread %d/%d!\n", bid,  gDim , tid, bDim);
}
int main(void)
{
    hello_from_gpu<<<2, 3>>>();
    cudaDeviceReset();
    return 0;
}