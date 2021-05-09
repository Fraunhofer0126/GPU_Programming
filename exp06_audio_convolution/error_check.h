#ifndef __ERROR_CHECK_H__
#define __ERROR_CHECK_H__

#define CHECK(error){ \
    const cudaError_t error_code = error; \
    if(error_code!=cudaSuccess){ \
        printf("####**CHECK**####\n"); \
        printf("line:%d in %s\n", __LINE__, __FILE__); \
        printf("Error needs to be handled!\n"); \
        printf("Error code:%d \n", error_code); \
        printf("Error string:%s \n\n", cudaGetErrorString(error_code)); \
    } \
} \


#endif

