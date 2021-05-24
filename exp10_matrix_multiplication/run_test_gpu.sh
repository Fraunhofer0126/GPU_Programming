nvcc -o mmg matrix_multiplication_gpu.cu
./mmg
./mmg test_data/matrix_M_512.txt test_data/matrix_N_512.txt _result_temp512.txt test_data/matrix_P_512.txt 
./mmg test_data/matrix_M_134_511.txt test_data/matrix_N_511_39.txt _result_temp511.txt test_data/matrix_P_134_39.txt 
