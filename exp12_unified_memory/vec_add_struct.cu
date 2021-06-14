#include<stdio.h>
#include<stdlib.h>
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
    Vector *d_vec1, *d_vec2, *d_vec_out;
    float *d_data1, *d_data2, *d_data_out;
    cudaMalloc((void**)&d_vec1, sizeof(Vector));
    cudaMalloc((void**)&d_data1, vec1->length * sizeof(float));
    cudaMemcpy(d_vec1, vec1, sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data1, vec1->data, vec1->length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_vec1->data), &d_data1, sizeof(float*), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_vec2, sizeof(Vector));
    cudaMalloc((void**)&d_data2, vec2->length * sizeof(float));
    cudaMemcpy(d_vec2, vec2, sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, vec2->data, vec2->length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_vec2->data), &d_data2, sizeof(float*), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_vec_out, sizeof(Vector));
    cudaMalloc((void**)&d_data_out, vec_out->length * sizeof(float));
    cudaMemcpy(d_vec_out, vec_out, sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_vec_out->data), &d_data_out, sizeof(float*), cudaMemcpyHostToDevice);
    
    kernel_vec_add<<<(vec_out->length-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(d_vec1, d_vec2, d_vec_out);

    //cudaMemcpy(&d_data_out, &(d_vec_out->data), sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(vec_out->data, d_data_out, vec_out->length * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(vec_out, d_vec_out, sizeof(Vector), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data_out);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vec_out);
}


int compare(float *in, float *ref, int length) {

    for (int i = 0; i < length/1000; i++)
    {
        float error = abs(in[i]-ref[i]);
        if(error>1e-2){
            printf("Results don't match! [%d] %f\n", i, error);
            return -1;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{   
    int test_size = 10000;
    Vector *vec1, *vec2, *vec_out, *vec_ref;

    vec1 = (Vector *)malloc(sizeof(Vector));
    vec2 = (Vector *)malloc(sizeof(Vector));
    vec_out = (Vector *)malloc(sizeof(Vector));
    vec_ref = (Vector *)malloc(sizeof(Vector));

    vec1->length = test_size;
    vec1->data = (float *)malloc(test_size*sizeof(float));

    vec2->length = test_size;
    vec2->data = (float *)malloc(test_size*sizeof(float));

    vec_out->length = test_size;
    vec_out->data = (float *)malloc(test_size*sizeof(float));

    vec_ref->length = test_size;
    vec_ref->data = (float *)malloc(test_size*sizeof(float));

    for(int i=0; i<test_size; i++)
    {
        vec1->data[i] = (float)i;
        vec2->data[i] = (float)2*i;
        vec_ref->data[i] = vec1->data[i] + vec2->data[i];
    }

    printf("Vec Add (GPU) %d:\n", test_size);

    gpu_vec_add(vec1, vec2, vec_out);

    if(compare(vec_out->data, vec_ref->data, test_size)==1){ printf("##Passed!\n\n"); }else{ printf("@@Failed!\n\n"); }

    free(vec1);
    free(vec2);
    free(vec_out);
    
    return 0;
}
