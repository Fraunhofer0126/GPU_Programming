#include<stdio.h>
__global__ void hello_from_gpu(){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int gdx = gridDim.x;
    int gdy = gridDim.y;
    int gdz = gridDim.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int bdz = blockDim.z;

    printf("Hello World from block-[%d/%d, %d/%d, %d/%d] and thread-[%d/%d, %d/%d, %d/%d]!\n", 
    bx, gdx, by, gdy, bz, gdz, tx, bdx, ty, bdy,  tz, bdz);
}
int main(void){
    dim3 block_size(2, 3, 4);
    dim3 grid_size(2, 1, 1);
    hello_from_gpu<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    printf("\nblock_size.x:%d, block_size.y:%d, block_size.z:%d", block_size.x, block_size.y, block_size.z);
    cudaDeviceReset();
    return 0;
}
