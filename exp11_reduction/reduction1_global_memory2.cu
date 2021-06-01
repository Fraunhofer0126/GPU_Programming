#include<stdio.h>
#include<stdlib.h>
#include"error_check.h"
#include"gpu_timer.h"

#define DTYPE double
#define DTYPE_FORMAT "%lf"
#define BLOCK_SIZE 32

// #define DTYPE float
// #define DTYPE_FORMAT "%f"

DTYPE partialSum(DTYPE *vector, int n) {
	DTYPE temp = 0;
	for (int i = 0; i < n; i++) {
		temp += vector[i];
	}
	return temp;
}

__global__ void kernel_reduction_non_consecutive(DTYPE *input, DTYPE *output, int n) {
	int tid = threadIdx.x, offset = blockIdx.x*blockDim.x;
	for(int s = 1; s < blockDim.x && tid*2 + s < BLOCK_SIZE; s<<=1){
		input[offset+tid*2] += input[offset+tid*2+s];
		__syncthreads();
	}
	if(tid == 0)
		output[blockIdx.x] = input[offset];	
}

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
 * Wrapper function that utilizes kernel function to sum the reduced results from blocks
*/
DTYPE gpu_reduction_gpu(DTYPE *input, int n,
                        void (*kernel)(DTYPE *input, DTYPE *output, int n))
{
	int MEM_SIZE = sizeof(DTYPE) * n;
	DTYPE *in = nullptr, *out = nullptr, *output = nullptr;

	CHECK(cudaMalloc((void**)&in, MEM_SIZE));
	CHECK(cudaMalloc((void**)&out, MEM_SIZE));
	output = (DTYPE*)malloc(MEM_SIZE);
	CHECK(cudaMemcpy(in, input, MEM_SIZE, cudaMemcpyHostToDevice));
	int grid = ceil((double)n/BLOCK_SIZE);
	//printf("grid = %d\n", grid);
	kernel<<<grid, BLOCK_SIZE>>>(in, out, n);
	
	CHECK(cudaMemcpy(output, out, MEM_SIZE, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(in));
	CHECK(cudaFree(out));
	/////////////////////////////////////////////////////////////////
	if(grid <= 32)
	{
		CHECK(cudaMalloc((void**)&in, MEM_SIZE));
		CHECK(cudaMalloc((void**)&out, MEM_SIZE));
		CHECK(cudaMemcpy(in, output, MEM_SIZE, cudaMemcpyHostToDevice));

		kernel<<<1, BLOCK_SIZE>>>(in, out, grid);

		CHECK(cudaMemcpy(output, out, MEM_SIZE, cudaMemcpyDeviceToHost));
		CHECK(cudaFree(in));
		CHECK(cudaFree(out));
	}
	else{
		for(; ; grid = ceil((double)grid/BLOCK_SIZE)){
			int temp_n = grid;
			//printf("grid = %d\n", grid);
			CHECK(cudaMalloc((void**)&in, MEM_SIZE));
			CHECK(cudaMalloc((void**)&out, MEM_SIZE));
			CHECK(cudaMemcpy(in, output, MEM_SIZE, cudaMemcpyHostToDevice));

			kernel<<<grid, BLOCK_SIZE>>>(in, out, temp_n);

			CHECK(cudaMemcpy(output, out, MEM_SIZE, cudaMemcpyDeviceToHost));
			CHECK(cudaFree(in));
			CHECK(cudaFree(out));
			if(grid < 32) break; //放在这里退出循环，确保grid<32但grid>1的时候还能在累加一次
		}
	}
	DTYPE ans = output[0];
	free(output);
	return ans;
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
	///cpu
	computed_result = partialSum(vector_input, n);
	///
	
	///gpu
	computed_result_gpu = reduction(vector_input, n, kernel);
	///


	printf("[%d] Computed sum (CPU): ", n);
	printf(DTYPE_FORMAT, computed_result);
	printf("  GPU result:");
	printf(DTYPE_FORMAT, computed_result_gpu);

	if (abs(computed_result_gpu - computed_result) < 1e-3) {
		printf("   PASSED! \n");
	} else {
		printf("   FAILED! \n");
	}
	printf("\n");

	free(vector_input);

}

int main(int argc, char **argv) {

	int n_arr[] = {1, 7, 585, 5000, 300001, 1<<20};
	for(int i=0; i<sizeof(n_arr)/sizeof(int); i++)
	{
		test(n_arr[i], gpu_reduction_gpu, kernel_reduction_non_consecutive);
		test(n_arr[i], gpu_reduction_gpu, kernel_reduction_consecutive);
	}

	return 0;
}
