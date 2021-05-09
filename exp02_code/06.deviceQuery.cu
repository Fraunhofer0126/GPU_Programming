#include<stdio.h>
#include<iostream>
using namespace std;
int main() {
    int dCount;
    cudaGetDeviceCount(&dCount);
    for(int i=0; i<dCount+3; i++)
    {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);
        if(err != cudaSuccess)
          cout<<"yes"<<endl;
        printf("CUDA Device#%d\n", i);
        printf("Device name:%s\n", prop.name);
        printf("multiProcessorCount:%d\n", prop.multiProcessorCount);
        printf("maxThreadsPerBlock:%d\n", prop.maxThreadsPerBlock);
        printf("warpSize:%d\n", prop.warpSize);
        printf("maxThreadsDim[3]:%d, %d, %d\n", 
        prop.maxThreadsDim[0], 
        prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
        printf("maxGridSize[3]:%d, %d, %d\n", 
        prop.maxGridSize[0], 
        prop.maxGridSize[1], 
        prop.maxGridSize[2]);
    }
    return 0;
}
