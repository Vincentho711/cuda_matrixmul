# Matrix Multiplication with CUDA
Implement a way to compute matrix multiplication with CUDA by computing sub-matrices in parallel with multiple threads. 2 different GPU kernels have been investigated here. The first one (matrixmul_kernel) involves utilising the multi-threading capability to compute the results using a standard matrix multiplication algorithm. The second one (cublas_kernel) utilises the optimised cublasSgemm() in the [cublas library](https://docs.nvidia.com/cuda/cublas/index.html). The performance of these 2 kernels has been compared to a baseline case which uses the CPU to to compute the standard matrix multiplication algorithm.

## Build
Navigate to the root directory of this project and run CMake to configure the project and generate a native build system. My platform is Windows and I configured it with
```sh
   C:\msys64\mingw64\bin\cmake.EXE --build c:/Users/XXX/cuda_projects/matrix_mul/build --config Debug --target ALL_BUILD -j 10 --
```
With linux, you can configure with
```sh
   cmake -S . -B build/
```
This will generate all the files in the `build` directory.

Then call that build system to actually compile/link the project
```sh
   cmake --build build/
```

Finally, call the executable with
```sh
   build/matrixmul
```




## Results
For the test below, two $1024 \times 1024$ square matrices have been multiplied together.
A NVidia GeForce GTX 1050 was used. 
| Method | Calculation time taken in ms (excluding memory allocation time) |
| --- | --- |
| Standard matrix multiplication with CPU | 6494 |
| Standard matrix multiplication with CUDA multi-threading | 23.01 |
| CUBLAS | 1.54 |