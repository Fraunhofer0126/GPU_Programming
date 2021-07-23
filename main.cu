#include<stdio.h>
#include<stdlib.h>
#define N 4
__global__
void func2(){ printf("GPU\n"); }
void func3(){ printf("CPU\n"); }
int main(void)
{
    func3();
    func2<<<1, 1>>>();
    cudaDeviceSynchronize();
}
