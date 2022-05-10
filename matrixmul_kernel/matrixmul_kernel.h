__global__ void maltrixmul_curand_init(unsigned int seed, curandState_t *states, int N);
__global__ void maltrixmul_init_matrices(curandState_t *states, int *a, int *b, int N);
__global__ void square_matrix_multiply(int *a, int *b, int*c, int N);
__global__ void square_matrix_multiply_shared(int *a, int *b, int *c, int N);