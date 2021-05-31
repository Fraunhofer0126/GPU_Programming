#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include<stdio.h>
#define CHECK(call){                                                    \
    cudaError_t e_code = call;                                          \
        if(e_code!=cudaSuccess){                                        \
            printf("##CUDA Error:\n");                                  \
            printf("  File: %s\n", __FILE__);                           \
            printf("  Line: %d\n", __LINE__);                           \
            printf("  Error info: %s\n", cudaGetErrorString(e_code));   \
        }                                                               \
}

#endif
