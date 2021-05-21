#include<stdio.h>
#include"error_check.h"
#include"gpu_timer.h"

#define DTYPE float
#define DTYPE_OUTPUT_FORMAT "%f "

#define H 1024
#define W 977
#define BLOCK_SIZE 32


void transpose_CPU(DTYPE *input, DTYPE *output, int num_rows, int num_cols)
{
	for(int row_idx=0; row_idx<num_rows; row_idx++)
	{
		for(int col_idx=0; col_idx<num_cols; col_idx++)
		{
			output[col_idx*num_rows+row_idx] = input[row_idx*num_cols+col_idx];
		}
	}
}

__global__ void kernel_transpose_serial(DTYPE *input, DTYPE *output, int num_rows, int num_cols)
{
	int input_width = num_cols;
	int output_width = num_rows;
	for(int row_idx=0; row_idx<num_rows; row_idx++)
	{
		for(int col_idx=0; col_idx<num_cols; col_idx++)
		{
			output[col_idx*output_width+row_idx] = input[row_idx*input_width+col_idx];
		}
	}
}

__global__ void kernel_transpose_per_row(DTYPE *input, DTYPE *output, int num_rows, int num_cols)
{
	int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int input_width = num_cols;
	int output_width = num_rows;
	for(int row_idx = 0; row_idx<num_rows; row_idx++)
	{
		if(row_idx<num_rows && col_idx<num_cols)
		{
			output[col_idx*output_width+row_idx] = input[row_idx*input_width+col_idx];
		}
	}
}

__global__ void kernel_transpose_per_element(DTYPE *input, DTYPE *output, int num_rows, int num_cols)
{
	int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

	int input_width = num_cols;
	int output_width = num_rows;

	if(row_idx<num_rows && col_idx<num_cols)
	{
		output[col_idx*output_width+row_idx] = input[row_idx*input_width+col_idx];
	}
}

/* 
* Todo:
* Implement the kernel function while satisfying the following requirements*
* 1.1 Utilizing shared memory to achieve coalesced memory access to both input and output matrices *
*/

__global__ void kernel_transpose_per_element_tiled(DTYPE *input, DTYPE *output, int num_rows, int num_cols)
{
	  int in_corner_i = blockIdx.x * BLOCK_SIZE;
	  int in_corner_j = blockIdx.y * BLOCK_SIZE;
	  int out_corner_i = blockIdx.y * BLOCK_SIZE;
	  int out_corner_j = blockIdx.x * BLOCK_SIZE;

	  int x = threadIdx.x, y = threadIdx.y;

	  __shared__ int tile[BLOCK_SIZE][BLOCK_SIZE];

	  tile[y][x] = input[(in_corner_i + x) + (in_corner_j + y) * num_cols];
	  __syncthreads();
	  output[(out_corner_i + x) + (out_corner_j + y) * num_rows] = tile[x][y];
}

/* 
* Todo:
* Implement the kernel function while satisfying the following requirements*
* 2.1 Utilizing shared memory to achieve coalesced memory access to both input and output matrices *
* 2.2 Avoid bank conflicts * 
*/
__global__ void kernel_transpose_per_element_tiled_no_bank_conflicts(DTYPE *input, DTYPE *output, int num_rows, int num_cols)
{
	  int in_corner_i = blockIdx.x * BLOCK_SIZE;
	  int in_corner_j = blockIdx.y * BLOCK_SIZE;
	  int out_corner_i = blockIdx.y * BLOCK_SIZE;
	  int out_corner_j = blockIdx.x * BLOCK_SIZE;

	  int x = threadIdx.x, y = threadIdx.y;

	  __shared__ int tile[BLOCK_SIZE+1][BLOCK_SIZE];

	  *(&tile[0][0]+y*BLOCK_SIZE+x+y) = input[(in_corner_i + x) + (in_corner_j + y) * num_cols];
	  __syncthreads();
	  output[(out_corner_i + x) + (out_corner_j + y) * num_rows] = *(&tile[0][0]+x*BLOCK_SIZE+y+x);
}


void init_data(DTYPE *arr, int n)
{
	for(int i=0; i<n; i++){
		arr[i] = (DTYPE)(i+1);
	}
}

int compare_matrices(DTYPE *input, DTYPE *ref, int num_rows, int num_cols)
{
	for(int row_idx=0; row_idx<num_rows; row_idx++)
	{
		for(int col_idx=0; col_idx<num_cols; col_idx++)
		{
			if(abs(ref[row_idx*num_cols+col_idx]-input[row_idx*num_cols+col_idx])>1e-3)
			{
				printf("Error:%f at (%d, %d)\n", abs(ref[row_idx*num_cols+col_idx]-input[row_idx*num_cols+col_idx]), row_idx, col_idx);
				return 0;
			}
		}
	}
	return 1;
}

 
int main() 
{
	int numBytes = H*W*sizeof(DTYPE);
	DTYPE *data_input = (DTYPE *)malloc(numBytes);
	DTYPE *data_output = (DTYPE *)malloc(numBytes);
	DTYPE *data_result = (DTYPE *)malloc(numBytes);

	init_data(data_input, H*W);
	transpose_CPU(data_input, data_result, H, W);


	DTYPE *d_in, *d_out;
	cudaMalloc((void **)&d_in, numBytes);
	cudaMalloc((void **)&d_out, numBytes);
	cudaMemcpy(d_in, data_input, numBytes, cudaMemcpyHostToDevice);
	GpuTimer timer;

	/* 
	* 1. matrix transpose serial *
	*/
	timer.Start();
	kernel_transpose_serial<<<1, 1>>>(d_in, d_out, H, W);
	timer.Stop();
	cudaMemcpy(data_output, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("\nTime cost (serial):%g ms. Veryifying results...%s\n",
			timer.Elapsed(), compare_matrices(data_output, data_result, H, W)?"Passed":"Failed");
	memset(data_output, 0, numBytes);
	cudaMemset(d_out, 0, numBytes);

	
	/* 
	* 2. matrix transpose per row *
	*/
	timer.Start();
	kernel_transpose_per_row<<<1, W>>>(d_in, d_out, H, W);
	timer.Stop();
	cudaMemcpy(data_output, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("\nTime cost (per row):%g ms. Veryifying results...%s\n",
			timer.Elapsed(), compare_matrices(data_output, data_result, H, W)?"Passed":"Failed");
	memset(data_output, 0, numBytes);
	cudaMemset(d_out, 0, numBytes);


	/* 
	* 3. matrix transpose per element *
	*/
	timer.Start();
	dim3 blocks((W-1)/BLOCK_SIZE+1, (H-1)/BLOCK_SIZE+1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	kernel_transpose_per_element<<<blocks, threads>>>(d_in, d_out, H, W);
	timer.Stop();
	cudaMemcpy(data_output, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("\nTime cost (per element):%g ms. Veryifying results...%s\n",
			timer.Elapsed(), compare_matrices(data_output, data_result, H, W)?"Passed":"Failed");
	memset(data_output, 0, numBytes);
	cudaMemset(d_out, 0, numBytes);


	/* 
	* 4. matrix transpose tiled with shared memory *
	*/
	timer.Start();
	kernel_transpose_per_element_tiled<<<blocks, threads>>>(d_in, d_out, H, W);
	timer.Stop();
	cudaMemcpy(data_output, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("\nTime cost (tiled with shared memory):%g ms. Veryifying results...%s\n",
			timer.Elapsed(), compare_matrices(data_output, data_result, H, W)?"Passed":"Failed");
	memset(data_output, 0, numBytes);
	cudaMemset(d_out, 0, numBytes);


	/* 
	* 5. matrix transpose tiled without bank conflicts *
	*/
	timer.Start();
	kernel_transpose_per_element_tiled_no_bank_conflicts<<<blocks, threads>>>(d_in, d_out, H, W);
	timer.Stop();
	cudaMemcpy(data_output, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("\nTime cost (tiled without bank conflicts):%g ms. Veryifying results...%s\n",
			timer.Elapsed(), compare_matrices(data_output, data_result, H, W)?"Passed":"Failed");



	free(data_input);
	free(data_output);
	free(data_result);
	cudaFree(d_in);
	cudaFree(d_out);

    return 0;
}
