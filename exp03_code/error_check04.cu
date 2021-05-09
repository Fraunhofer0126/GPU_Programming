#include"error_check.h"
#include<stdio.h>
int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 1));
    return 0;
}
