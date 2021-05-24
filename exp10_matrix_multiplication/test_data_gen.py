import numpy as np
import os

def write_w_to_file(matrix, filename):
    with open(filename, 'w') as f:
        f.write(' '.join([str(x) for x in matrix.shape]))
        f.write('\n')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                f.write(str(matrix[i, j])+" ")
            f.write('\n')

if __name__ == '__main__':
    target_dir= "test_data"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    m = 512
    n = 512
    p = 512
    w1 = np.random.randn(m, n) * 255
    w2 = np.random.randn(n, p) * 0.001
    write_w_to_file(w1, os.path.join(target_dir, "matrix_M_512.txt"))
    write_w_to_file(w2, os.path.join(target_dir, "matrix_N_512.txt"))
    write_w_to_file(w1.dot(w2), os.path.join(target_dir, "matrix_P_512.txt"))


    m = 134
    n = 511
    p = 39
    w1 = np.random.randn(m, n) * 255
    w2 = np.random.randn(n, p) * 0.001
    write_w_to_file(w1, os.path.join(target_dir, "matrix_M_134_511.txt"))
    write_w_to_file(w2, os.path.join(target_dir, "matrix_N_511_39.txt"))
    write_w_to_file(w1.dot(w2), os.path.join(target_dir, "matrix_P_134_39.txt"))