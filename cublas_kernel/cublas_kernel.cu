#ifndef CUBLAS_KERNEL_H
#define CUBLAS_KERNEL_H

#include <cublas_v2.h>
#include <stdlib.h>
#include "cublas_kernel.h"

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