#include<stdio.h>
__global__ void hello(){
    printf("*");
}

int main() {
    cudaError_t error_code;
    hello<<<-1, 1>>>();

    error_code = cudaGetLastError();
    printf("%d\n", error_code);
    if(error_code!=cudaSuccess){
        printf("\n");
        printf("line:%d in %s\n", __LINE__, __FILE__);
        printf("Error needs to be handled!\n");
        printf("Error code:%d \n", error_code);
        printf("Error string:%s \n", cudaGetErrorString(error_code));
    }

    hello<<<1, 1025>>>();
    error_code = cudaGetLastError();
    if(error_code!=cudaSuccess){
        printf("\n");
        printf("line:%d in %s\n", __LINE__, __FILE__);
        printf("Error needs to be handled!\n");
        printf("Error code:%d \n", error_code);
        printf("Error string:%s \n", cudaGetErrorString(error_code));
    }

    error_code = cudaDeviceSynchronize();
    if(error_code!=cudaSuccess){
        printf("\n");
        printf("line:%d in %s\n", __LINE__, __FILE__);
        printf("Error needs to be handled!\n");
        printf("Error code:%d \n", error_code);
        printf("Error string:%s \n", cudaGetErrorString(error_code));
    }

    error_code  = cudaDeviceReset();
    if(error_code!=cudaSuccess){
        printf("\n");
        printf("line:%d in %s\n", __LINE__, __FILE__);
        printf("Error needs to be handled!\n");
        printf("Error code:%d \n", error_code);
        printf("Error string:%s \n", cudaGetErrorString(error_code));
    }


    return 0;
}
