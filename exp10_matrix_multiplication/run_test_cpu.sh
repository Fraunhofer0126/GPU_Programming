#!/bin/sh
nvcc -o mmc matrix_multiplication_cpu.cu
./mmc
./mmc test_data/matrix_M_512.txt test_data/matrix_N_512.txt _result_temp512.txt test_data/matrix_P_512.txt 
./mmc test_data/matrix_M_134_511.txt test_data/matrix_N_511_39.txt _result_temp511.txt test_data/matrix_P_134_39.txt 
