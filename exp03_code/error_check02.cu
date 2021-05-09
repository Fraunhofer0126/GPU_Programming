#include<stdio.h>

int main() {

    cudaDeviceProp prop;

    int device_id = 0;
    printf("\nGet properties from device #%d:\n", device_id);
    cudaError_t error_code = cudaGetDeviceProperties(&prop, device_id);
    if(error_code==cudaSuccess)
    {
        printf("CUDA API successed!\n");
    }
    else if(error_code==cudaErrorInvalidDevice)
    {
        printf("Invalid Device! code:%d \n", error_code);
    }

    device_id = 1;
    printf("\nGet properties from device #%d:\n", device_id);
    error_code = cudaGetDeviceProperties(&prop, device_id);
    if(error_code==cudaSuccess)
    {
        printf("CUDA API successed!\n");
    }
    else if(error_code==cudaErrorInvalidDevice)
    {
        printf("Invalid Device! code:%d \n", error_code);
        printf("line:%d in %s\n", __LINE__, __FILE__);
    }
    return 0;
}
