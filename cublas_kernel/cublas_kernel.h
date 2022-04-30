__global__ void cublas_curand_init(unsigned int seed, curandState_t *states, int N);
__global__ void cublas_init_matrices(curandState_t *states, int *a, int *b, int N);
void cublas_matrix_multiply(cublasHandle_t &handle, int *a, int *b, int*c, int N);