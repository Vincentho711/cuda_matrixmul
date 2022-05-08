#include <iostream>
#include <math.h>
#include <cstdlib>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cxxopts.hpp>
#include "matrixmul_kernel/matrixmul_kernel.h"
#include "cublas_kernel/cublas_kernel.h"
#include "config.h"

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

void run_maltrix_kernal_test(int N)
{
    std::cout << "Starting matrix mul kernel test..." << std::endl;
    
    size_t bytes = N*N*sizeof(int);
    // Allocate unified memory for the matrices
    int *a, *b, *c, *d;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    cudaMallocManaged(&d, bytes);

    // Setup grid and block size
    int threads = 16;
    int blocks = (N + threads -1)/ threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Initialise time counter for CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Object for curand
    curandState_t *states;

    // Allocate space on the GPU for the random states 
    cudaMallocManaged((void**) &states, N * N * sizeof(curandState_t));

    // Initialise matrices on the GPU
    std::cout << "Initialise matrices" << std::endl;
    maltrixmul_curand_init<<<BLOCKS, THREADS>>>(time(0), states, N); 
    maltrixmul_init_matrices<<<BLOCKS,THREADS>>>(states, a, b, N);
    // loop through the array elements
    /* for ( int i = 0; i < 5; i++)
    {
        std::cout << *(a + i) << ", ";
    } */
    // Launch the kernel
    std::cout << "Launching kernel" << std::endl;
    cudaEventRecord(start);
    square_matrix_multiply<<<BLOCKS,THREADS>>>(a, b, c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Multiplication with matrix_mul kernel (GPU) completed."<< std::endl;
    std::cout << "matrix_mul kernel took: " ;
    std::cout << milliseconds ;
    std::cout << " ms" << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaFree(states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_cublas_kernel_test(int N)
{
    std::cout << "Starting cublas kernel test..." << std::endl;

    size_t bytes = N*N*sizeof(int);
    // Allocate unified memory for the matrices
    int *a, *b, *c, *d;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    cudaMallocManaged(&d, bytes);

    // Setup grid and block size
    int threads = 16;
    int blocks = (N + threads -1)/ threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Initialise time counter for CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Object for curand
    curandState_t *states;

    // Allocate space on the GPU for the random states 
    cudaMalloc((void**) &states, N * N * sizeof(curandState_t));

    // Initialise matrices on the GPU 
    std::cout << "Initialise matrices" << std::endl;
    cublas_curand_init<<<BLOCKS,THREADS>>>(time(0), states, N); 
    cublas_init_matrices<<<BLOCKS,THREADS>>>(states, a, b, N);
    // std::cout << *a << std::endl;

    // Launch the kernel
    std::cout << "Launching kernel" << std::endl;
    cudaEventRecord(start);
    cublas_matrix_multiply(handle, a, b, c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Multiplication with cublas kernel (GPU) completed."<< std::endl;
    std::cout << "GPU took: " ;
    std::cout << milliseconds;
    std::cout << " ms" << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaFree(states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

void run_std_matrixmul(int N)
{
    size_t bytes = N*N*sizeof(int);
    // Allocate memory for the matrices on CPU
    int *a, *b, *c, *d;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    cudaMallocManaged(&d, bytes);

    // Initialise our matrices
    init_matrix(a, N);
    init_matrix(b, N);
    

    // Compute standard method for matrix multiplcation
    auto start = std::chrono::high_resolution_clock::now();
    std_matrixmul(a, b, c, N);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Multiplication with CPU completed."<< std::endl;
    std::cout << "CPU took: " ;
    std::cout << duration.count() ;
    std::cout << " ms" << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
}

int main(int argc, char *argv[])
{
    // Instantiate cxxopts object
    cxxopts::Options options("matrix_mul", "Matrix Multiplication on the GPU with Nvidia CUDA.");

    options.add_options()
        ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false")) // a bool parameter
        ("i,integer", "Int param", cxxopts::value<int>())
        ("f,file", "File name", cxxopts::value<std::string>())
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    std::cout << argv[0] << " Version " << MATRIXMUL_VERSION_MAJOR << "." << MATRIXMUL_VERSION_MINOR << std::endl;
    std::cout << "Debug: " << result["debug"].as<bool>() << std::endl;
    if (result.count("integer"))
    {
        std::cout << "Integer: " << result["integer"].as<int>() << std::endl;
    }

    // Set N first
    int N = 1<<10;

    // Lauch the standard maltrixmul test
    run_std_matrixmul(N);
    
    // Launch the matrixmul kernel test
    run_maltrix_kernal_test(N);

    // Launch the cublas kernel test
    run_cublas_kernel_test(N);
    
    return 0;
}