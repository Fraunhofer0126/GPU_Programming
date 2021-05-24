#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

void print_matrix(float *matrix, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; row++)
	{
		for (int col = 0; col < num_cols; col++)
		{
			printf("%f ", matrix[row*(num_cols)+col]);
		}
        printf("\n");
    }
}

int compare_matrix(float *in_matrix, float *ref_matrix, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; row++)
	{
		for (int col = 0; col < num_cols; col++)
		{
            float error = abs(in_matrix[row*(num_cols)+col]-ref_matrix[row*(num_cols)+col]);
            if(error>1e-2){
                printf("Results don't match! [%d, %d] %f", row, col, error);
                return -1;
            }
		}
    }
    return 1;
}

float* read_matrix(const char *filename, int *num_rows, int *num_cols) {
    FILE *fpread;
    fpread = fopen(filename, "r");
    if (fpread == NULL){
        printf("Error when reading file %s\n", filename);
        exit(-1);
    }
	
    fscanf(fpread, "%d %d", num_rows, num_cols);
    float *matrix = (float *)malloc((*num_rows) * (*num_cols) * sizeof(float));
    for (int row = 0; row < *num_rows; row++)
	{
		for (int col = 0; col < *num_cols; col++)
		{
			fscanf(fpread, "%f", &matrix[row*(*num_cols)+col]);
		}
    }
    
    fclose(fpread);
    printf("Load matrix from %s (rows: %d, columns: %d)\n", filename, *num_rows, *num_cols);
    return matrix;
}

void write_matrix(const char *filename, float *matrix, int num_rows, int num_cols) {
    FILE *fpwrite;
    fpwrite = fopen(filename, "w");
    if (fpwrite == NULL){
        printf("Error when writing file %s\n", filename);
        exit(-1);
    }
	
    fprintf(fpwrite, "%d %d\n", num_rows, num_cols);

    for (int row = 0; row < num_rows; row++)
	{
		for (int col = 0; col < num_cols; col++)
		{
			fprintf(fpwrite, "%f ", matrix[row*num_cols+col]);
		}
        fprintf(fpwrite, "\n");
    }
    
    fclose(fpwrite);
    printf("Store matrix to %s (rows: %d, columns: %d)\n", filename, num_rows, num_cols);
}

#endif