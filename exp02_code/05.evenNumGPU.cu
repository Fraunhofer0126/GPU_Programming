#include<stdio.h>

__global__ void evenNum_gpu()
{
    //int tid = threadIdx.x;
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid%2==0)
    {
        printf("Even number: %d\n", tid);
    }
}

int main() {
    int numUpperBound = 10;
    printf("\nEven numbers less than %d (GPU version):\n", numUpperBound);
    evenNum_gpu<<<3, numUpperBound>>>();
    cudaDeviceReset();
    return 0;
}
