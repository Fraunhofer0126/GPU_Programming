#include"error_check.h"
#include<stdio.h>
__global__ void hello(){
    printf("Hello World!\n");
}

int main() {
    hello<<<-1, 1>>>();
    CHECK(cudaGetLastError());
    

    hello<<<1, 1025>>>();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());   

    CHECK(cudaDeviceReset());
    return 0;
}
