#include"error_check.h"
#include"text_helper.h"
#include<stdio.h>

__global__ void decrypt_gpu(char *d_encryptedStr, char *d_decryptedStr, int lenStr, int pwd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<lenStr)
    {
        d_decryptedStr[idx] = d_encryptedStr[idx] - (idx%pwd+1);
    }
}

void print_msg(char *input, int n)
{
    for(int i=0; i<n; i++)
    {
        printf("%c", input[i]);
    }
}

int main(int argc, char *argv[])
{
    if(argc!=4)
    {
        printf("Usage: command  input-text-file-path  output-text-file-path password");
        return -1;
    }  
    const char *input_file = argv[1];   // "input.txt";
    const char *output_file = argv[2];   // "output.txt";
    const int pwd = atoi(argv[3]);

    printf("\nReading content from %s ...\n", input_file);
    int string_size, read_size;
    char *inputStr = ReadFile(input_file, &read_size, &string_size);
    int lenStr = read_size+1;
    
    //ToDo
    char *outputStr = (char*)malloc(sizeof(char)*lenStr);
    char *d_enStr, *d_deStr;
    cudaMalloc((void**)&d_enStr, lenStr);
    cudaMalloc((void**)&d_deStr, lenStr);
    cudaMemcpy(d_deStr, inputStr, lenStr, cudaMemcpyHostToDevice);
    decrypt_gpu<<<ceil(lenStr/1024.0), 1024>>>(d_deStr, d_enStr, lenStr, pwd);
    cudaMemcpy(outputStr, d_enStr, lenStr, cudaMemcpyDeviceToHost);
    cudaFree(d_enStr);
    cudaFree(d_deStr);
    // Write to output file
    WriteFile(output_file, outputStr, read_size);

    return 0;
}
