#ifndef MATRIXMUL_KERNEL_H
#define MATRIXMUL_KERNEL_H

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "matrixmul_kernel.h"

// Initialise states for random number generation
__global__ void maltrixmul_curand_init(unsigned int seed, curandState_t *states, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    // Associate a squence number to values in the states array
    // The sequence number (second parameter) should be different for each state
    curand_init(seed, (row*N + col) , 0, &states[row*N+col]);
}

// Initialise matrices directly on GPU for performance
__global__ void maltrixmul_init_matrices(curandState_t *states, int *a, int *b, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Generate value in a, b with different sequence numbers
    if (row < N && col < N)
    {
        a[row*N + col] = curand(&states[row*N + col]) % 100;
        b[row*N + col] = curand(&states[row*N + col]) % 100;
    }
    
}
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