#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#include"error_check.h"
#include"time_helper.h"
#include<iostream>
using namespace std;
void rgb_to_blur_cpu(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int BLUR_SIZE){
    for(int row=0; row<height; row++){
        for(int col=0; col<width; col++){
            int pixVal1 = 0;
            int pixVal2 = 0;
            int pixVal3 = 0;
            int pixels = 0;
            int offset = (row * width + col)*channels;
            for(int blurrow = -BLUR_SIZE; blurrow <= BLUR_SIZE; ++blurrow){
                for(int blurcol = -BLUR_SIZE; blurcol <= BLUR_SIZE; ++blurcol){
                    int currow = row + blurrow;
                    int curcol = col + blurcol;
                    if(currow > -1 && currow < height && curcol > -1 && curcol < width){
                        pixVal1 += input_image[(currow * width + curcol) * channels];
                        pixVal2 += input_image[(currow * width + curcol) * channels + 1];
                        pixVal3 += input_image[(currow * width + curcol) * channels + 2];
                        pixels++;
                    }
                }
            }
            *(output_image + offset) = (unsigned char)(pixVal1 / pixels);
            *(output_image + offset + 1) = (unsigned char)(pixVal2 / pixels);
            *(output_image + offset + 2) = (unsigned char)(pixVal3 / pixels);
            if(channels==4)
            {
                *(output_image + offset + 3) = input_image[offset + 3];
            }
        }
    }
}
__global__
void blur_gpu(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int blur_size)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if(Col < width && Row < height){
        int pixValr = 0, pixValg = 0, pixValb = 0, pixels = 0;
        for(int i = -blur_size; i <= blur_size; i++){
            for(int j = -blur_size; j <= blur_size; j++){
                int curRow = Row + i;
                int curCol = Col + j;
                int offset = (curRow*width+curCol)*channels;
                if(curRow >= 0 && curRow < height && curCol >=0 && curCol < width){
                    pixValr += input_image[offset];
                    pixValg += input_image[offset+1];
                    pixValb += input_image[offset+2];
                    pixels++;
                }
            }
        }
        output_image[(Row * width + Col)*channels] = (unsigned char)(pixValr/pixels);
        output_image[(Row * width + Col)*channels+1] = (unsigned char)(pixValg/pixels);
        output_image[(Row * width + Col)*channels+2] = (unsigned char)(pixValb/pixels);
    }
}
int main(int argc, char *argv[])
{
    if(argc<6)
    {
        printf("Usage: command input-image-name  output-image-name  cpu/gpu? channels  blursize");
        return -1;
    }
    char *input_image_name = argv[1];
    char *output_image_name = argv[2];
    char *option = argv[3];
    char *channel = argv[4];
    char *bsize = argv[5];
    int blur_size = 0;
    for(int i = 0; i < strlen(bsize); i++){
        blur_size *= 10;
        blur_size += *(bsize+i)-'0';
    }
    int desired_no_channels = *channel-'0';
    int width, height, original_no_channels;
    
    unsigned char *input_img = stbi_load(input_image_name, &width, &height, &original_no_channels, desired_no_channels);
    if(input_img==NULL){ printf("Error in loading the image.\n"); exit(1);}
    printf("Loaded image with a width of %dpx, a height of %dpx. The original image had %d channels, the loaded image has %d channels.\n", width, height, original_no_channels, desired_no_channels);
    int channels = original_no_channels;
    int img_mem_size = width * height * channels * sizeof(char);
    double begin;
    if(strcmp(option, "cpu")==0)
    {
        printf("Processing with CPU!\n");
        unsigned char *sepia_img = (unsigned char *)malloc(img_mem_size);
        if(sepia_img==NULL){  printf("Unable to allocate memory for the sepia image. \n");  exit(1);  }
        begin = cpuSecond();
        rgb_to_blur_cpu(input_img, sepia_img, width, height, channels, blur_size);
        printf("Time cost [CPU]:%f s\n", cpuSecond()-begin);
        stbi_write_jpg(output_image_name, width, height, channels, sepia_img, 100);
        free(sepia_img);
    }
    else if(strcmp(option, "gpu")==0) 
    {
        printf("Processing with GPU!\n");
        unsigned char *d_input_img, *d_output_img;
        cudaMalloc((void**)&d_input_img, img_mem_size);
        cudaMalloc((void**)&d_output_img, img_mem_size);
        unsigned char *output_img = (unsigned char *)malloc(img_mem_size);
        cudaMemcpy(d_input_img, input_img, img_mem_size, cudaMemcpyHostToDevice);
        dim3 block(64, 64, 1);
        dim3 grid((width-1)/block.x+1, (height-1)/block.y+1, 1);
        begin = cpuSecond();
        blur_gpu<<<block, grid>>>(d_input_img, d_output_img, width, height, channels, blur_size);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        printf("Time cost [GPU]:%f s\n", cpuSecond()-begin);
        cudaMemcpy(output_img, d_output_img, img_mem_size, cudaMemcpyDeviceToHost);
        stbi_write_jpg(output_image_name, width, height, channels, output_img, 100);
        free(output_img);
        cudaFree(d_input_img);
        cudaFree(d_output_img);
    }
    stbi_image_free(input_img);
    return 0;
}