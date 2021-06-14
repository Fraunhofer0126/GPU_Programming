#include<stdio.h>
#include<stdlib.h>
#include"gpu_timer.h"
#include"error_check.h"

#define BLOCK_SIZE 128

struct Vector {
    float *data;
    int length;
};


__global__ void kernel_vec_add(Vector *vec1, Vector *vec2, Vector *vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < vec1->length)
    {
        vec_out->data[idx] = vec1->data[idx] + vec2->data[idx];
    }       
}

// Todo
// Implement the following function to wrap the vector add kernel function
void gpu_vec_add(Vector *vec1, Vector *vec2, Vector *vec_out)
{
    kernel_vec_add<<<(vec_out->length-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(vec1, vec2, vec_out);
    cudaDeviceSynchronize();
}

int compare(float *in, float *ref, int length) {

    for (int i = 0; i < length; i++)
    {
        float error = abs(in[i]-ref[i]);
        if(error>1e-2){
            printf("Results don't match! [%d] %f", i, error);
            return -1;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{   
    int test_size = 10000;
    Vector *vec1, *vec2, *vec_out, *vec_ref;

    cudaMallocManaged((void **)&vec1, sizeof(Vector));
    cudaMallocManaged((void **)&(vec1->data), test_size*sizeof(float));
    vec1->length = test_size;

    cudaMallocManaged((void **)&vec2, sizeof(Vector));
    cudaMallocManaged((void **)&(vec2->data), test_size*sizeof(float));
    vec2->length = test_size;

    cudaMallocManaged((void **)&vec_out, sizeof(Vector));
    cudaMallocManaged((void **)&(vec_out->data), test_size*sizeof(float));
    vec_out->length = test_size;

    cudaMallocManaged((void **)&vec_ref, sizeof(Vector));
    cudaMallocManaged((void **)&(vec_ref->data), test_size*sizeof(float));
    vec_ref->length = test_size;

    for(int i=0; i<test_size; i++)
    {
        vec1->data[i] = (float)i;
        vec2->data[i] = (float)2*i;
        vec_ref->data[i] = vec1->data[i] + vec2->data[i];
    }

    printf("Vec Add (GPU) %d:\n", test_size);

    gpu_vec_add(vec1, vec2, vec_out);

    if(compare(vec_out->data, vec_ref->data, test_size)==1){ printf("##Passed!\n\n"); }else{ printf("@@Failed!\n\n"); }

    cudaFree(vec1);
    cudaFree(vec2);
    cudaFree(vec_out);
    cudaFree(vec_ref);
    
    return 0;
}
