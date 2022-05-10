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
__global__ void square_matrix_multiply(int *a, int *b, int *c, int N){
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
__global__ void square_matrix_multiply_shared(int *a, int *b, int *c, int N)
{
    // Set up a tile as 16 x 16
    const int tile_size = 16;
    // Calculate the memory the tile uses in bytes
    const int mem_size = tile_size*tile_size*sizeof(int);
    
    __shared__ int aTile[mem_size];
    __shared__ int bTile[mem_size];
    // aTile pointer to the start of shared memory s
    // int *aTile = s;
    // bTile pointer starts at the Nth index of s
    // int *bTile = (int *)&aTile[N];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    for(int i = 0; i < (N / tile_size); i++)
    {
        // Load one element into shared memory per thread in a threadblock
        // N / tile_size is the number of threadblocks required to iterate through the whole matrices

        // For matrix a, row*N indexes the global row for this thread (loop invariant),
        // tile_size indexes the next set of columns each iteration
        // threadIdx.x indexes the specific column within the set of columns
        aTile[(threadIdx.y*tile_size) + threadIdx.x] = a[row*N + (tile_size*i + threadIdx.x)];

        // For matrix b, col indexes the global volumn (loop invariant)
        // i*tile_size*N indexes the next set of rows each iteration
        // threadIdx.y* indexes the specific row within the set of rows 
        bTile[(threadIdx.y*tile_size) + threadIdx.x] = b[col + (i*tile_size*N + threadIdx.y*N)];

        __syncthreads();

        // For aTile and bTile, sweep across the elements within a tile, multiply and aggregate
        for (int j = 0; j < tile_size; j++)
        {
            temp_sum += aTile[(threadIdx.y*tile_size) + j] * bTile[(j*tile_size) + threadIdx.x];
        }
        __syncthreads();
    }
    c[(row*N + col)] = temp_sum;
}
#endif