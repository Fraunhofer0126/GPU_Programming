#include<stdio.h>
#include "error_check.h"
__global__ void func1(int x)
{
    int tid = threadIdx.x;
    printf("thread: %d, parameter:%d  \n", tid, x);
}
 
__global__ void func2(int *x)
{
    int tid = threadIdx.x;
    printf("thread: %d, parameter:%d  \n", tid, *x);
}
 
__global__ void func3(int x[], int n)
{
    int tid = threadIdx.x;
    printf("thread: %d, parameter:%d  \n", tid, x[tid]);
}
int main(){
    const int gx = 2, bx = 2;
    int a = 42, aa = 88;
    int *b = &aa;
    int c[] = {1, 1, 2, 3, 5, 8};


    func1<<<gx, bx>>>(a);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    

    func2<<<gx, bx>>>(b);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    

    func3<<<gx, bx>>>(c, 6);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    
    cudaDeviceReset();
    return 0;
}
