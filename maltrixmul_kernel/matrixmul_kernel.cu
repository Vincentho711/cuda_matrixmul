#ifndef MATRIXMUL_KERNEL_H
#define MATRIXMUL_KERNEL_H

#include <stdio.h>
#include "matrixmul_kernel.h"

__global__ void square_matrix_multiply(int *a, int *b, int*c, int N){
    // Figure out base and offset for rows and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check whether threads are in bounds of the matrix
    if (row < N && col < N)
    {
        int temp_sum = 0;
        // Each thread goes through one row and one col
        for (int i = 0;i < N; i++)
        {
            temp_sum += a[row*N + i]*b[col + N*i];
        }
        // write the result for each
        c[row*N + col] = temp_sum;

    } 
}
#endif