#include <iostream>
#include <math.h>
#include <cstdlib>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cublas_v2.h>
#include "maltrixmul_kernel/matrixmul_kernel.h"
#include "cublas_kernel/cublas_kernel.h"

// Initialise a square matrix with random numbers from 0 to 100
void init_matrix(int *m, int N)
{
    for (int i = 0; i<N*N; i++)
    {
        m[i] = rand() % 100;

    }

}

// Standard matrix multiplication for performance comparison
void std_matrixmul(int *a, int *b, int *c, int N)
{
    // Rudimentary matrix multiplication
    for (int i = 0; i < N ; i++)
    {
        for (int k = 0; k < N ; k++)
        {
            for (int j = 0; j < N ; j++)
            {
                c[i*N+j] += a[i*N+k]*b[j+k*N];
            }
        }
    }
}

/*
// Verify matrix multiplication on the CPU
void verify_multiply_results(int *a, int *b, int *c, int N)
{
    // Rudimentary matrix multiplication
    for (int i = 0; i < N ; i++)
    {
        int temp_sum;
        for (int j = 0; j < N ; j++)
        {
            temp_sum = 0;
            for (int k = 0; k < N ; k++)
            {
                temp_sum += a[i*N+k]*b[j+k*N];
            }
            // Check each result
            assert(temp_sum == c[i*N+j]);
        }
    }
}
*/

int main(){

    // Multiplication
    // Set up the matrices to be operated on
    // For a matrix with dimension NxN
    // Set N first
    int N = 1<<10;

    size_t bytes = N*N*sizeof(int);
    // Allocate memory for the matrices
    int *a, *b, *c, *d;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    cudaMallocManaged(&d, bytes);
    
    // Initialise our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Setup grid and block size
    int threads = 16;
    int blocks = (N + threads -1)/ threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Launch the kernel
    auto start = std::chrono::high_resolution_clock::now();
    square_matrix_multiply<<<BLOCKS,THREADS>>>(a, b, c, N);
    auto stop = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Multiplication with matrix_mul kernel (GPU) completed."<< std::endl;
    std::cout << "GPU took: " ;
    std::cout << duration.count() ;
    std::cout << " ms" << std::endl;

    // Launch the cublas kernel
    start = std::chrono::high_resolution_clock::now();
    cublas_matrix_multiply(handle, a, b, c, N);
    stop = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Multiplication with cublas kernel (GPU) completed."<< std::endl;
    std::cout << "GPU took: " ;
    std::cout << duration.count() ;
    std::cout << " ms" << std::endl;
    
    // Compute standard method for matrix multiplcation
    start = std::chrono::high_resolution_clock::now();
    std_matrixmul(a, b, c, N);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Multiplication with CPU completed."<< std::endl;
    std::cout << "CPU took: " ;
    std::cout << duration.count() ;
    std::cout << " ms" << std::endl;

    //std::cout << "GPU results" << std::endl;
    //std::cout << *c << std::endl;
    //std::cout << "CPU results" << std::endl;
    //std::cout << *d << std::endl;

}