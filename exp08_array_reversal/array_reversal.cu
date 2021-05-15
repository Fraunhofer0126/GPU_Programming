#include<stdio.h>
#include"error_check.h"

#define DTYPE int
#define DTYPE_OUTPUT_FORMAT "%d "

#define BLOCK_SIZE 32

__global__ void kernel_reverse_naive(DTYPE *input, DTYPE *output, int n)
{
    int idx_in = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_out = n - idx_in - 1;
    if(idx_in<n)
    {
		output[idx_out] = input[idx_in];
    }
}

/* 
* Todo:
* Implement the following kernel function *
* 1. Utilizing shared memory to achieve sequential access to input data *
*/
__global__ void kernel_reverse_shared_mem(DTYPE *input, DTYPE *output, int n)
{
	int idx_in = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ DTYPE s_data[BLOCK_SIZE];
	s_data[BLOCK_SIZE - threadIdx.x - 1] = input[idx_in];
	__syncthreads();
	if(idx_in < n){
		output[threadIdx.x+(gridDim.x-blockIdx.x-1)*BLOCK_SIZE] = s_data[threadIdx.x];
	}
}

void reverse_gpu(DTYPE *data_input, DTYPE *data_output, int n, void (*kernel)(DTYPE *input, DTYPE *output, int n))
{
	DTYPE *d_data_input = NULL;
	DTYPE *d_data_output = NULL;
	CHECK(cudaMalloc((void **)&d_data_input, n*sizeof(DTYPE)));
	CHECK(cudaMalloc((void **)&d_data_output, n*sizeof(DTYPE)));
	CHECK(cudaMemcpy(d_data_input, data_input, n*sizeof(DTYPE), cudaMemcpyHostToDevice));
	CHECK(cudaMemset(d_data_output, 0, n*sizeof(DTYPE)));

	int blockDim = BLOCK_SIZE;
	int gridDim = (n-1)/blockDim + 1;
	kernel<<<gridDim, blockDim>>>(d_data_input, d_data_output, n);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(data_output, d_data_output, n*sizeof(DTYPE), cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_data_input));
	CHECK(cudaFree(d_data_output));
}

void reverse_cpu(DTYPE *data_input, DTYPE *data_output, int n)
{
	for(int i=0; i<n; i++)
	{
		data_output[n-i-1] = data_input[i];
	}
}

void init_data(DTYPE *arr, int n)
{
	for(int i=0; i<n; i++){
		arr[i] = (i+1);
	}
}

void print_arr(DTYPE *arr, int n){
	for(int i=0; i<n; i++){
		printf(DTYPE_OUTPUT_FORMAT, arr[i]);
	}
}

int compare(DTYPE *arr, DTYPE *ref, int n){
	for(int i=0; i<n; i++){
		if(abs(arr[i]-ref[i])>1){
			return -1;
		}
	}
	return 1;
}

int main() 
{
	int n = 1<<6;

	DTYPE *data_input = (DTYPE *)malloc(n*sizeof(DTYPE));
	DTYPE *data_output_ref = (DTYPE *)malloc(n*sizeof(DTYPE));
	DTYPE *data_output_from_gpu = (DTYPE *)malloc(n*sizeof(DTYPE));
	
	init_data(data_input, n);
	memset(data_output_ref, 0, n*sizeof(DTYPE));
	reverse_cpu(data_input, data_output_ref, n);

	printf("\nInput array:\n");
	print_arr(data_input, n);
	printf("\n Ref array:\n");
	print_arr(data_output_ref, n);

	printf("\n***   sanity check   ***\n");

	/*
	***  Check 1 ***
	*/
	reverse_gpu(data_input, data_output_from_gpu, n, kernel_reverse_naive);

	print_arr(data_output_from_gpu, n);
	printf("\n [array_reversal_naive]: ");
	if(compare(data_output_from_gpu, data_output_ref, n)==1){ printf("Passed!\n");	}else{ printf("Failed!\n");	}
	memset(data_output_from_gpu, 0, n*sizeof(DTYPE));


	/*
	***  Check 2  ***
	*/
	reverse_gpu(data_input, data_output_from_gpu, n, kernel_reverse_shared_mem);

	print_arr(data_output_from_gpu, n);
	printf("\n [array_reversal_shared_memory]: ");
	if(compare(data_output_from_gpu, data_output_ref, n)==1){ printf("Passed!\n");	}else{ printf("Failed!\n");	}
	memset(data_output_from_gpu, 0, n*sizeof(DTYPE));



	/*
	###  Reverse a long array ###
	*/
	n = 1<<26;
	printf("\n\n***   Reversing a long array [length: %d]    ***\n", n);
	DTYPE *data_input_long = (DTYPE *)malloc(n*sizeof(DTYPE));
	DTYPE *data_output_long_ref = (DTYPE *)malloc(n*sizeof(DTYPE));
	DTYPE *data_output_long_from_gpu = (DTYPE *)malloc(n*sizeof(DTYPE));
	init_data(data_input_long, n);
	memset(data_output_long_ref, 0, n*sizeof(DTYPE));
	reverse_cpu(data_input_long, data_output_long_ref, n);

	/*
	***  GPU Implementations 1 ***
	*/
	float time_cost_gpu = -1;
	cudaEvent_t gpu_start, gpu_stop;

	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start);

	reverse_gpu(data_input_long, data_output_long_from_gpu, n, kernel_reverse_naive);
	
	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&time_cost_gpu, gpu_start, gpu_stop);

	printf("\n [array_reversal_naive] Time cost (GPU):%f ms ", time_cost_gpu);
	printf("\n [array_reversal_naive] (GPU): ");
	if(compare(data_output_long_from_gpu, data_output_long_ref, n)==1){ printf("Passed!\n");	}else{ printf("Failed!\n");	}
	memset(data_output_long_from_gpu, 0, n*sizeof(DTYPE));


	/*
	***  GPU Implementations 2 ***
	*/
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start);

	reverse_gpu(data_input_long, data_output_long_from_gpu, n, kernel_reverse_shared_mem);

	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&time_cost_gpu, gpu_start, gpu_stop);

	printf("\n [kernel_reverse_shared_memory] Time cost (GPU):%f ms ", time_cost_gpu);
	printf("\n [kernel_reverse_shared_memory] (GPU): ");
	if(compare(data_output_long_from_gpu, data_output_long_ref, n)==1){ printf("Passed!\n");	}else{ printf("Failed!\n");	}


	cudaDeviceReset();
    return 0;
}
