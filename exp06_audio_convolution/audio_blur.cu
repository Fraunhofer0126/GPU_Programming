#include<stdio.h>
#include<math.h>
#include<sndfile.h>
#include"error_check.h"
#include"time_helper.h"

const int GAUSSIAN_SIDE_WIDTH = 10;
const int GAUSSIAN_SIZE = 2*GAUSSIAN_SIDE_WIDTH + 1;
const float PI = 3.14159265358979;

float gaussian(const int x, const float mu, const float sigma)
{
	
	float factor1 = sigma*sqrt(2*PI);
	float factor2 = pow((x-mu), 2);
	float factor3 = 2*pow(sigma, 2);
	float result = (1/factor1) * exp(-1.0 * (factor2 / factor3));
	return result;
}

float *gaussian_filter(const float mu, const float sigma)
{
	
	float *filter = (float *)malloc(GAUSSIAN_SIZE*sizeof(float));
	for(int i=-GAUSSIAN_SIDE_WIDTH; i<=GAUSSIAN_SIDE_WIDTH; i++)
	{
		filter[i+GAUSSIAN_SIDE_WIDTH] = gaussian(i, mu, sigma);
	}

	float total = 0.0;
	for(int i=0; i<GAUSSIAN_SIZE; i++)
	{
		total += filter[i];
	}

	for(int i=0; i<GAUSSIAN_SIZE; i++)
	{
		filter[i] /= total;
	}
	return filter;
}

void h_convolution(float *single_channel_output,
		const float *single_channel_input, const long n_frames,
		const float *filter, const int filter_size)
{
	for (int i = 0; i < filter_size; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			single_channel_output[i] += single_channel_input[i-j]
									* filter[j];
		}
	}
	for (int i = filter_size; i < n_frames; i++)
	{
		for (int j = 0; j < filter_size; j++)
		{
			single_channel_output[i] += single_channel_input[i-j]
									* filter[j];
		}
	}
}

__global__ void setting_channels_gpu(float *single_channel_input, float *all_channel_input, int ch, int n_channels, int n_frames){
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	if(i < n_frames){
		single_channel_input[i] = all_channel_input[(i*n_channels)+ch];
	}
}

__global__ void h_convolution_gpu(float *single_channel_output,
	const float *single_channel_input, const long n_frames,
	const float *filter, const int filter_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	if(i<filter_size){
		for(int j=0; j<=i ;j++){
			single_channel_output[i] += single_channel_input[i-j] * filter[j];
		}
	}else if(i<n_frames){
		for (int j = 0; j < filter_size; j++)
		{
			single_channel_output[i] += single_channel_input[i-j] * filter[j];
		}
	}else{
		return;
	}
}

__global__ void h_convolution_gpu_shared(float *single_channel_output,
	const float *single_channel_input, const long n_frames,
	const float *filter, const int filter_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	extern __shared__ float s_filter[];
	if(threadIdx.x<filter_size) s_filter[threadIdx.x] = filter[threadIdx.x];
	__syncthreads();
	if(i<filter_size){
		for(int j=0; j<=i ;j++){
			single_channel_output[i] += single_channel_input[i-j] * s_filter[j];
		}
	}else if(i<n_frames){
		for (int j = 0; j < filter_size; j++)
		{
			single_channel_output[i] += single_channel_input[i-j] * s_filter[j];
		}
	}else{
		return;
	}
}

__global__ void channel_signleToall(float *all_channel_output, float *single_channel_output,int n_frames, int n_channels, int ch){
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	if(i<n_frames){
		all_channel_output[(i*n_channels)+ch] = single_channel_output[i];
	}
}



int main(int argc, char *argv[])
{
	if(argc!=5)
	{
		printf("Arguments: \n<input file name> \n<cpu output file name> \n<gpu global output file name> \n<gpu shared output file name>\n");
		return -1;
	}
	const char *input_file_name = argv[1]; // "test.wav"; // "example_test.wav";
	const char *output_file_name = argv[2];  // "output.wav"; // "example_test.wav";
	const char *output_file_name_gpu = argv[3];
	const char *output_file_name_gpu_shared = argv[4];
	SNDFILE *in_file_handle;
	SF_INFO in_file_info;
	
	int amt_read;
	printf("Reading %s ...\n", input_file_name);
	in_file_handle = sf_open(input_file_name, SFM_READ, &in_file_info);
	if(!in_file_handle)
	{
		printf("Open file failed!");
		exit(1);
	}	
	long n_frames = in_file_info.frames;
	int n_channels = in_file_info.channels;

	size_t data_size = sizeof(float)*n_frames*n_channels;
	float *all_channel_input = (float *)malloc(data_size);
	amt_read = sf_read_float(in_file_handle, all_channel_input, n_frames*n_channels);
	// assert(amt_read == in_file_info.frames*in_file_info.channels);
	if(amt_read != n_frames*n_channels)
	{
		printf("Error occurs during file reading! \n");
	}

	printf("n_frames:%ld, n_channels:%d\n", n_frames, n_channels);
	sf_close(in_file_handle);
	
	
	/* Filters */
	int mu = 0;
	int sigma = 5;
	float *filter = gaussian_filter(mu, sigma);

	/* GPU_Filters */
	float *filter_gpu = NULL;
	cudaMalloc((void **)&filter_gpu, GAUSSIAN_SIZE*sizeof(float));
	cudaMemcpy(filter_gpu, filter, GAUSSIAN_SIZE*sizeof(float), cudaMemcpyHostToDevice);

	printf("\n Convolution ... \n");
	// Split Channels
	float *all_channel_output = (float *)malloc(data_size);
	
	float *single_channel_input = (float *)malloc(n_frames*sizeof(float));
	float *single_channel_output = (float *)malloc(n_frames*sizeof(float));

	float *all_channel_input_gpu = NULL;
	float *all_channel_output_gpu = NULL;
	float *single_channel_input_gpu = NULL;
	float *single_channel_output_gpu = NULL;
	cudaMalloc((void**)&all_channel_input_gpu, data_size);
	cudaMemcpy(all_channel_input_gpu, all_channel_input, data_size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&all_channel_output_gpu, data_size);
	cudaMalloc((void**)&single_channel_input_gpu, n_frames*sizeof(float));
	cudaMalloc((void**)&single_channel_output_gpu, n_frames*sizeof(float));

	float *all_channel_input_gpu_shared = NULL;
	float *all_channel_output_gpu_shared = NULL;
	float *single_channel_input_gpu_shared = NULL;
	float *single_channel_output_gpu_shared = NULL;
	cudaMalloc((void**)&all_channel_input_gpu_shared, data_size);
	cudaMemcpy(all_channel_input_gpu_shared, all_channel_input, data_size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&all_channel_output_gpu_shared, data_size);
	cudaMalloc((void**)&single_channel_input_gpu_shared, n_frames*sizeof(float));
	cudaMalloc((void**)&single_channel_output_gpu_shared, n_frames*sizeof(float));

	float gpu_total_time_global = 0, gpu_total_time_shared = 0;
	double cpu_time = 0;
	cudaEvent_t start, stop;
	float elapsedTime;
	int blockDim = 128;
	int gridDim = (n_frames-1)/blockDim + 1;
	for(int ch=0; ch<n_channels; ch++)
	{
		printf("Processing channel %d ... \n", ch);
		for(int i=0; i<n_frames; i++)
		{
			single_channel_input[i] = all_channel_input[(i*n_channels)+ch];
		}
		memset(single_channel_output, 0, n_frames*sizeof(float));

		double begin, time_cost;
		begin = cpuSecond();
		// Convolution using CPU		
		h_convolution(single_channel_output,
				single_channel_input, n_frames,
				filter, GAUSSIAN_SIZE);	
		time_cost = cpuSecond() - begin;
		cpu_time += time_cost;
		for(int i=0; i<n_frames; i++)
		{
			all_channel_output[(i*n_channels)+ch] = single_channel_output[i];
		}
		printf("CPU Time Cost : %.9lf s\n", time_cost);

		setting_channels_gpu<<<gridDim, blockDim>>>(single_channel_input_gpu, all_channel_input_gpu, ch, n_channels, n_frames);
		cudaMemset(single_channel_output_gpu, 0, n_frames*sizeof(float));
		CHECK(cudaDeviceSynchronize());
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		h_convolution_gpu<<<gridDim, blockDim>>>(single_channel_output_gpu,
				single_channel_input_gpu, n_frames, filter_gpu, GAUSSIAN_SIZE);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("GPU Global Time Cost: %.9f s\n",elapsedTime/1000);
		gpu_total_time_global += elapsedTime;
		CHECK(cudaDeviceSynchronize());
		channel_signleToall<<<gridDim, blockDim>>>(all_channel_output_gpu, single_channel_output_gpu, n_frames ,n_channels, ch);

		setting_channels_gpu<<<gridDim, blockDim>>>(single_channel_input_gpu_shared, all_channel_input_gpu_shared, ch, n_channels, n_frames);
		cudaMemset(single_channel_output_gpu_shared, 0, n_frames*sizeof(float));
		CHECK(cudaDeviceSynchronize());
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		h_convolution_gpu_shared<<<gridDim, blockDim, GAUSSIAN_SIZE*sizeof(float)>>>(single_channel_output_gpu_shared,
				single_channel_input_gpu_shared, n_frames, filter_gpu, GAUSSIAN_SIZE);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("GPU Shared Time Cost: %.9f s\n",elapsedTime/1000);
		gpu_total_time_shared += elapsedTime;
		CHECK(cudaDeviceSynchronize());
		channel_signleToall<<<gridDim, blockDim>>>(all_channel_output_gpu_shared, single_channel_output_gpu_shared, n_frames ,n_channels, ch);
		printf("\n");
	}

	printf("CPU Total Time : %.9lf s\n",cpu_time);
	printf("GPU Global Total Time : %.9f s\n",gpu_total_time_global/1000);
	printf("GPU Shared Total Time : %.9f s\n",gpu_total_time_shared/1000);
	printf("\n\n");

	// Write to file
	SNDFILE *out_file_handle;
	SF_INFO out_file_info;
	
	out_file_info = in_file_info;
	out_file_handle = sf_open(output_file_name, SFM_WRITE, &out_file_info);
	if(!out_file_handle)
	{
		printf("Output failed!");
		exit(1);
	}
	sf_write_float(out_file_handle, all_channel_output, amt_read);
	sf_close(out_file_handle);
	free(all_channel_input);
	free(all_channel_output);
	free(single_channel_input);
	free(single_channel_output);

	float *all_channel_output_gpu_in_host = (float *)malloc(data_size);
	cudaMemcpy(all_channel_output_gpu_in_host, all_channel_output_gpu, data_size, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	SNDFILE *out_file_handle_gpu;
	out_file_handle_gpu = sf_open(output_file_name_gpu, SFM_WRITE, &out_file_info);
	if(!out_file_handle_gpu)
	{
		printf("Output failed!");
		exit(1);
	}
	sf_write_float(out_file_handle_gpu, all_channel_output_gpu_in_host, amt_read);
	sf_close(out_file_handle_gpu);

	cudaFree(all_channel_input_gpu);
	cudaFree(all_channel_output_gpu);
	cudaFree(single_channel_input_gpu);
	cudaFree(single_channel_output_gpu);
	free(all_channel_output_gpu_in_host);

	float *all_channel_output_gpu_shared_in_host = (float *)malloc(data_size);
	cudaMemcpy(all_channel_output_gpu_shared_in_host, all_channel_output_gpu_shared, data_size, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	SNDFILE *out_file_handle_gpu_shared;
	out_file_handle_gpu_shared = sf_open(output_file_name_gpu_shared, SFM_WRITE, &out_file_info);
	if(!out_file_handle_gpu_shared)
	{
		printf("Output failed!");
		exit(1);
	}
	sf_write_float(out_file_handle_gpu_shared, all_channel_output_gpu_shared_in_host, amt_read);
	sf_close(out_file_handle_gpu_shared);

	printf("Results have been saved to %s , %s , %s \n", output_file_name, output_file_name_gpu, output_file_name_gpu_shared);

	cudaFree(all_channel_input_gpu_shared);
	cudaFree(all_channel_output_gpu_shared);
	cudaFree(single_channel_input_gpu_shared);
	cudaFree(single_channel_output_gpu_shared);
	free(all_channel_output_gpu_shared_in_host);

	return 0;
}
