#include<stdio.h>
#include"error_check.h"

#define DTYPE char
#define _MAXIMUM 32
#define BLOCK_SIZE 32

__global__ void kernel_offset(DTYPE *input, DTYPE *output, int offset)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  output[idx] = input[idx+offset] + 1;
}

__global__ void kernel_stride(DTYPE *input, DTYPE *output, int stride)
{
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
  output[idx] = input[idx] + 1;
}

void run_test(const char *param_name, void (*kernel)(DTYPE *input, DTYPE *output, int param))
{
	int nMB = 4;
	int n = nMB * 1024 * 1024 / sizeof(DTYPE);
	DTYPE *d_data_input = NULL;
	DTYPE *d_data_output = NULL;
	CHECK(cudaMalloc((void **)&d_data_input, (_MAXIMUM+1)*n*sizeof(DTYPE))); // avoid illegal mem access in offset and strided case
	CHECK(cudaMalloc((void **)&d_data_output, (_MAXIMUM+1)*n*sizeof(DTYPE)));

	float ms;
	cudaEvent_t startEvent, stopEvent;
	CHECK(cudaEventCreate(&startEvent));
  	CHECK(cudaEventCreate(&stopEvent));

	int blockDim = BLOCK_SIZE;
	int gridDim = (n-1)/blockDim + 1;
	printf("%s, Bandwidth (GB/s):\n", param_name);
	for(int i=0; i<=_MAXIMUM; i++)
	{	
		CHECK(cudaEventRecord(startEvent, 0);)

		kernel<<<gridDim, blockDim>>>(d_data_input, d_data_output, i);

		CHECK(cudaEventRecord(stopEvent, 0);)
    	CHECK(cudaEventSynchronize(stopEvent);)
    	CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent);)
		printf("%d,   %f\n", i, 2*nMB/ms);  // DDR, doubled rate
	}

	CHECK(cudaFree(d_data_input));
	CHECK(cudaFree(d_data_output));
}


int main() 
{
	printf("Test: global memory access with offset! \n");
	run_test("Offset", kernel_offset);

	printf("\nTest: global memory access with stride! \n");
	run_test("Stride", kernel_stride);

	cudaDeviceReset();
    return 0;
}
