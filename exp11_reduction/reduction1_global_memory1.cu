#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include"error_check.h"
#include"gpu_timer.h"

#define DTYPE double
#define DTYPE_FORMAT "%lf"
#define BLOCK_SIZE 32

float time_cost_gpu = -1, time_cost_cpu = -1;
cudaEvent_t gpu_start, gpu_stop, cpu_start, cpu_stop;

/* CPU implementation */
DTYPE partialSum(DTYPE *vector, int n) {
	DTYPE temp = 0;
	for (int i = 0; i < n; i++) {
		temp += vector[i];
	}
	return temp;
}

/*
 * Todo:
 * reduction kernel in which the threads are mapped to data with stride 2
*/
__global__ void kernel_reduction_non_consecutive(DTYPE *input, DTYPE *output, int n) {
	int tid = threadIdx.x, offset = blockIdx.x*blockDim.x;
	for(int s = 1; s < blockDim.x && tid*2 + s < BLOCK_SIZE; s<<=1){ //主要防无关thread多加
		input[offset+tid*2] += input[offset+tid*2+s];
		__syncthreads();
	}
	if(tid == 0)
		output[blockIdx.x] = input[offset];	
}

/*
 * Todo:
 * reduction kernel in which the threads are consecutively mapped to data
*/
__global__ void kernel_reduction_consecutive(DTYPE *input, DTYPE *output, int n) {
	int tid = threadIdx.x, offset = blockIdx.x*blockDim.x;
	for(int s = BLOCK_SIZE/2; s >= 1 && tid+s < BLOCK_SIZE; s>>=1){
		input[offset+tid] += input[offset+tid+s];
		__syncthreads();
	}
	if(tid == 0)
		output[blockIdx.x] = input[offset];
}

/*
 * Todo:
 * Wrapper function that utilizes cpu computation to sum the reduced results from blocks
*/
DTYPE gpu_reduction_cpu(DTYPE *input, int n,
		void (*kernel)(DTYPE *input, DTYPE *output, int n)) {
	int MEM_SIZE = sizeof(DTYPE) * n;
	DTYPE *in = nullptr, *out = nullptr, *output = nullptr;

	CHECK(cudaMalloc((void**)&in, MEM_SIZE));
	CHECK(cudaMalloc((void**)&out, MEM_SIZE));
	output = (DTYPE*)malloc(MEM_SIZE);
	CHECK(cudaMemcpy(in, input, MEM_SIZE, cudaMemcpyHostToDevice));
	int grid = ceil((double)n/BLOCK_SIZE);

	cudaEventRecord(gpu_start);
	kernel<<<grid, BLOCK_SIZE>>>(in, out, n);
	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&time_cost_gpu, gpu_start, gpu_stop);
	
	CHECK(cudaMemcpy(output, out, MEM_SIZE, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(in));
	CHECK(cudaFree(out));
	
	DTYPE sum = 0;
	for(int i = 0; i < grid; i += 1){
		sum += output[i];
	} 
	free(output);
	return sum;
}


DTYPE* test_data_gen(int n) {
	srand(time(0));
	DTYPE *data = (DTYPE *) malloc(n * sizeof(DTYPE));
	for (int i = 0; i < n; i++) {
		data[i] = 1.0 * (rand() % RAND_MAX) / RAND_MAX;
	}
	return data;
}

void test(int n,
		DTYPE (*reduction)(DTYPE *input, int n,
		                        void (*kernel)(DTYPE *input, DTYPE *output, int n)),
		                        void (*kernel)(DTYPE *input, DTYPE *output, int n))
{
	DTYPE computed_result, computed_result_gpu;
	DTYPE *vector_input;
	vector_input = test_data_gen(n);

	printf("---------------------------\n");


	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventCreate(&cpu_start);
	cudaEventCreate(&cpu_stop);

	///cpu
	cudaEventRecord(cpu_start);
	computed_result = partialSum(vector_input, n);
	cudaEventRecord(cpu_stop);
	cudaEventSynchronize(cpu_stop);
	cudaEventElapsedTime(&time_cost_cpu, cpu_start, cpu_stop);
	printf("Time cost (CPU):%f ms \n", time_cost_cpu);
	///
	
	///gpu
	computed_result_gpu = reduction(vector_input, n, kernel);
	printf("Time cost (GPU):%f ms \n", time_cost_gpu);
	///
	printf("[%d] Computed sum (CPU): ", n);
	printf(DTYPE_FORMAT, computed_result);
	printf("  GPU result:");
	printf(DTYPE_FORMAT, computed_result_gpu);

	if (abs(computed_result_gpu - computed_result) < 1e-3) {
		printf("  PASSED! \n");
	} else {
		printf("  FAILED! \n");
	}
	printf("\n");

	free(vector_input);

}

int main(int argc, char **argv) {

	int n_arr[] = {1, 7, 585, 5000, 300001, 1<<20};
	for(int i=0; i<sizeof(n_arr)/sizeof(int); i++)
	{
		test(n_arr[i], gpu_reduction_cpu, kernel_reduction_non_consecutive);
		test(n_arr[i], gpu_reduction_cpu, kernel_reduction_consecutive);
	}

	return 0;
}