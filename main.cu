#include<stdio.h>
#include<stdlib.h>
#define N 4

int main(void)
{
    int arr[N][N] = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}, {13,14,15,16}};
    printf("Original 2D array: \n");
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }

    printf("\n Row-major layout: \n");
 
    int *p1 = NULL;
    int *p2 = NULL;
    int *p3 = NULL;
    // Todo
    p1 = arr[0];
    printf("Approach 1: (address:%p)\n", p1);   
    for(int j=0; j<N*N; j++){
        printf("%d ", *(p1+j));
    }
    printf("\n");
    // Todo
    p2 = *arr;
    printf("Approach 2: (address:%p)\n", p2);
    for(int j=0; j<N*N; j++){
        printf("%d ", *(p2+j));
    }
    printf("\n");
    // Todo
    p3 = &arr[0][0];
    printf("Approach 3: (address:%p)\n", p3);
    for(int j=0; j<N*N; j++){
        printf("%d ", *(p3+j));
    }

    return 0;
}
