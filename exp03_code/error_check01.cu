#include<stdio.h>

int main() {
    cudaError_t error_code = cudaDeviceReset();
    printf("returned error code:%d \n", error_code); // a cudaError_t variable can be regarded as a integer
    printf("cudaSuccess:%d, error_code==cudaSuccess:%d \n", cudaSuccess, cudaSuccess==error_code);
    if(error_code==cudaSuccess)
    {
        printf("CUDA API successed!\n");
    }
    else
    {
        printf("Error needs to be handled! code:%d \n", error_code);
    }
    return 0;
}
