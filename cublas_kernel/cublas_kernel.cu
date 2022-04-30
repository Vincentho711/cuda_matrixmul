#ifndef CUBLAS_KERNEL_H
#define CUBLAS_KERNEL_H

#include <cublas_v2.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_kernel.h"

// Initialise states for random number generation
__global__ void cublas_curand_init(unsigned int seed, curandState_t *states, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    // Associate a squence number to values in the states array
    // The sequence number (second parameter) should be different for each state
    curand_init(seed, (row*N + col) , 0, &states[row*N+col]);
}

// Initialise matrices directly on GPU for performance
__global__ void cublas_init_matrices(curandState_t *states, int *a, int *b, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Generate value in a, b with different sequence numbers
    if (row < N && col < N)
    {
        a[row*N + col] = curand(&states[row*N + col]) % 100;
        b[row*N + col] = curand(&states[row*N + col]) % 100;
    }

}

void cublas_matrix_multiply(cublasHandle_t &handle, int *a, int *b, int *c, int N){
    // Scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Calculate c = (alpha*a) * b + (beta*c)
    // (m X n) * (n X k ) = (m X k)
    // Signature : handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, (float*)a, N, (float*)b, N, &beta, (float*)c, N); 
}
#endif