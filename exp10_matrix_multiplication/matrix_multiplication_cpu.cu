#include<stdio.h>
#include<stdlib.h>
#include"matrix_helper.h"
#include"time_helper.h"

#define BLOCK_SIZE 16

void matrix_multiply(const float *M, const float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols)
{
    for(int row=0; row<M_rows; row++){
        for(int col=0; col<N_cols; col++){
            float Pvalue = 0;
            for(int k=0; k<M_cols; k++){
                Pvalue += M[row*M_cols+k] * N[k*N_cols+col];
            }
            P[row*N_cols+col] = Pvalue;
        }
    }
}

/* 
* Todo:
* Implement the following block matrix multiplication function *
*/
void matrix_multiply_block(const float *M, const float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols)
{
    for(int b_row = 0; b_row < M_rows; b_row += BLOCK_SIZE){
        for(int b_col = 0; b_col < N_cols; b_col += BLOCK_SIZE){
            for(int b_k = 0; b_k < M_cols; b_k += BLOCK_SIZE){
                //
                for(int row = b_row; row < b_row + BLOCK_SIZE && row < M_rows; row++){
                    for(int col = b_col; col < b_col + BLOCK_SIZE && col < N_cols; col++){
                        double Pvalue = P[row*N_cols+col]; //Not `Pvalue = 0`.
                        for(int k = b_k; k < b_k + BLOCK_SIZE && k < M_cols; k++){
                            Pvalue += M[row*M_cols+k]*N[k*N_cols+col];
                        }
                        P[row*N_cols+col] = Pvalue;  //if Pvalue=0 below, use "+="" here.
                    }
                }
            }
        }
    }

}

int main(int argc, char **argv)
{   
    double begin;

    if(argc==1){
        printf("###-----------------begin(argc==1)-----------------###\n");
        int test_size = 1024;
        float *M1 = (float *)malloc(test_size * test_size * sizeof(float));
        float *N1 = (float *)malloc(test_size * test_size * sizeof(float));
        printf("Matrix Multiplication (CPU):\n");
        float *P1 = (float *)malloc(test_size * test_size * sizeof(float));
        float *Pb1 = (float *)malloc(test_size * test_size * sizeof(float));

        begin = cpuSecond();
        matrix_multiply(M1, N1, P1, test_size, test_size, test_size, test_size);
        printf("Time cost:%lf s\n", cpuSecond()-begin);

        begin = cpuSecond();
        matrix_multiply_block(M1, N1, Pb1, test_size, test_size, test_size, test_size);
        printf("Time cost:%lf s\n", cpuSecond()-begin);

        if(compare_matrix(P1, Pb1, test_size, test_size)==1){ printf("##Passed!\n\n"); }else{ printf("@@Failed!\n\n"); }
        free(M1);
        free(N1);
        free(P1);
        free(Pb1);
        exit(-1);
        printf("###-----------------end(argc==1)-----------------###\n");
    }

    if(argc<4) {
        printf("Usage: command    matrix-M-filename    matrix-N-filename   matrix-P-filename   matrix-R-filename");
        return -1;
    }
    const char *matrix_M_filename = argv[1];
    const char *matrix_N_filename = argv[2];
    const char *matrix_P_filename = argv[3];
    const char *matrix_R_filename = argv[4];
    
    printf("###-----------------begin(argc>1)-----------------###\n");
    /* 
    *  Load M matrix *
    */
    int M_rows, M_cols;
    float *M = read_matrix(matrix_M_filename, &M_rows, &M_cols);
    // print_matrix(M, M_rows, M_cols);
    
    
    /* 
    *  Load N matrix *
    */
    int N_rows, N_cols;
    float *N = read_matrix(matrix_N_filename, &N_rows, &N_cols);
    // print_matrix(N, N_rows, N_cols);

    if(M_cols!=N_rows){
        printf("Rows and Cols don't match!");
        exit(-1);
    }
    
    /* 
    *  Matrix Multiplication  *
    */
    printf("Matrix multiplication (CPU) \n");
    float *P = (float *)malloc(M_rows * N_cols * sizeof(float));
    begin = cpuSecond();
    matrix_multiply(M, N, P, M_rows, M_cols, N_rows, N_cols);
    printf("Time cost:%lf s\n", cpuSecond()-begin);
    

    /* 
    *  Block Matrix Multiplication  *
    */
    printf("Block Matrix Multiplication (CPU) \n");
    float *Pb = (float *)malloc(M_rows * N_cols * sizeof(float));
    begin = cpuSecond();
    matrix_multiply_block(M, N, Pb, M_rows, M_cols, N_rows, N_cols);
    printf("Time cost:%lf s\n", cpuSecond()-begin);

    if(compare_matrix(P, Pb, M_rows, N_cols)==1){ printf("Passed!\n"); }else{ printf("Failed!\n"); }


   /* 
    *  Save Results  *
    */
    write_matrix(matrix_P_filename, Pb, M_rows, N_cols);
    
    printf("---------------------------\n");
    /* 
    *  Check Results  *
    */
    if(matrix_R_filename!=NULL){
        printf("\n Checking results ... \n");
        int R_rows, R_cols;
        float *R = read_matrix(matrix_R_filename, &R_rows, &R_cols);;
        if(compare_matrix(P, R, R_rows, R_cols)==1){ printf("Passed!\n"); }else{ printf("Failed!\n"); }
        if(compare_matrix(Pb, R, R_rows, R_cols)==1){ printf("Passed!\n"); }else{ printf("Failed!\n"); }
        free(R);
    }
    printf("###-----------------end(argc>1)-----------------###\n");
    
    free(M);
    free(N);
    free(P);
    free(Pb);
    
    return 0;
}
