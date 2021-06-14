#include <math.h>
#include <stdio.h>
#include "error_check.h"
#include "gpu_timer.h"

#define BLOCK_SIZE 128

typedef double DTYPE;

const int N = 38400; 
const int NUM_STREAMS = 30;

void vec_add_cpu(const DTYPE *h_x, const DTYPE *h_y, DTYPE *h_z, const int n)
{
    for (int i=0; i < n; i++)
    {
        h_z[i] = h_x[i] + h_y[i];
    }
}

int vec_compare(const DTYPE *h_x, const DTYPE *h_y, const int n)
{
    for(int x=0; x<n; x++){
        if(abs(h_x[x]-h_y[x])>1e-3){
            printf("Results don't match! [%d] [%f - %f]\n", x, h_y[x], h_x[x]);
            return -1;
        }            
    }
    return 1;
}

void __global__ vec_add_kernel(const DTYPE *d_x, const DTYPE *d_y, DTYPE *d_z, const int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        // repeat to increase the computational cost
        for(int x=0; x<1000000; x++)
        { 
                d_z[idx] = d_x[idx] + d_y[idx];
        }
    }
}

void vec_add_default_stream(const DTYPE *h_x, const DTYPE *h_y, DTYPE *h_z, const int n);

void vec_add_multiple_streams_overlapped(const DTYPE *h_x, const DTYPE *h_y, DTYPE *h_z, const int n);


int main(void)
{
    DTYPE *h_x, *h_y, *h_z;
    // Todo 1
    // Allocate host memory for pointers [*h_x, *h_y, *h_z] using cudaMallocHost
    cudaMallocHost((void**)&h_x, N*sizeof(DTYPE), cudaHostAllocDefault);
    cudaMallocHost((void**)&h_y, N*sizeof(DTYPE), cudaHostAllocDefault);
    cudaMallocHost((void**)&h_z, N*sizeof(DTYPE), cudaHostAllocDefault);
     
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }
    
    vec_add_cpu(h_x, h_y, h_z, N);
    
     vec_add_default_stream(h_x, h_y, h_z, N);
        
    vec_add_multiple_streams_overlapped(h_x, h_y, h_z, N);


    // Todo 2
    // Free host memory pointers [*h_x, *h_y, *h_z] using cudaFreeHost
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_z);

    return 0;
}


void vec_add_default_stream(const DTYPE *h_x, const DTYPE *h_y, DTYPE *h_z, const int n)
{
    DTYPE *d_x, *d_y, *d_z;
    DTYPE *h_z1 = (DTYPE*) malloc(sizeof(DTYPE) * N);
    
    CHECK(cudaMalloc(&d_x, sizeof(DTYPE) * N));
    CHECK(cudaMalloc(&d_y, sizeof(DTYPE) * N));
    CHECK(cudaMalloc(&d_z, sizeof(DTYPE) * N));
    CHECK(cudaMemcpy(d_x, h_x, sizeof(DTYPE) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(DTYPE) * N, cudaMemcpyHostToDevice));

    const int grid_size = (n - 1) / BLOCK_SIZE + 1;
    
    GpuTimer timer;
    timer.Start();
    vec_add_kernel<<<grid_size, BLOCK_SIZE>>>(d_x, d_y, d_z, n);
    CHECK(cudaMemcpy(h_z1, d_z, sizeof(DTYPE) * N, cudaMemcpyDeviceToHost));
    timer.Stop();
    
    printf("[vec_add_default_stream] Time cost: %f ms\n", timer.Elapsed());     
    CHECK(cudaDeviceSynchronize());
    if(vec_compare(h_z1, h_z, N)==1){ printf("  PASSED!\n");  }else{  printf("  FAILED\n");  }
    
    free(h_z1);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));    
}


// Todo 3
// @kangan
// Using multiple streams to tmplement the following function achieve overlapped memcpy [cudaMemcpyAsync] and kernel computing
void vec_add_multiple_streams_overlapped(const DTYPE *h_x, const DTYPE *h_y, DTYPE *h_z, const int n)
{
    cudaStream_t stream[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++) cudaStreamCreate(&stream[i]);
    DTYPE *d_x, *d_y, *d_z, *h_z1 = (DTYPE*) malloc(sizeof(DTYPE) * N);
    CHECK(cudaMalloc(&d_x, sizeof(DTYPE) * N));
    CHECK(cudaMalloc(&d_y, sizeof(DTYPE) * N));
    CHECK(cudaMalloc(&d_z, sizeof(DTYPE) * N));

    const int grid_size = (n / NUM_STREAMS - 1) / BLOCK_SIZE + 1;

    float time = 0;
    for(int i = 0; i < NUM_STREAMS; i++){
        CHECK(cudaMemcpyAsync(d_x+i*(N/NUM_STREAMS), h_x+i*(N/NUM_STREAMS), sizeof(DTYPE) * (N/NUM_STREAMS), cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(d_y+i*(N/NUM_STREAMS), h_y+i*(N/NUM_STREAMS), sizeof(DTYPE) * (N/NUM_STREAMS), cudaMemcpyHostToDevice, stream[i]));
        GpuTimer timer;
        timer.Start();
        vec_add_kernel<<<grid_size, BLOCK_SIZE, 0, stream[i]>>>(d_x+i*N/NUM_STREAMS, d_y+i*N/NUM_STREAMS, d_z+i*N/NUM_STREAMS, n);
        timer.Stop();
        time += timer.Elapsed();
        CHECK(cudaMemcpyAsync(h_z1+i*(N/NUM_STREAMS), d_z+i*(N/NUM_STREAMS), sizeof(DTYPE) * (N/NUM_STREAMS), cudaMemcpyDeviceToHost, stream[i]));
    }
    
    printf("[vec_add_multiple_stream] Time cost: %f ms\n", time);     
    CHECK(cudaDeviceSynchronize());
    if(vec_compare(h_z1, h_z, N)==1){ printf("  PASSED!\n");  }else{  printf("  FAILED\n");  }
    
    free(h_z1);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y)); 
    CHECK(cudaFree(d_z));    
    
}
