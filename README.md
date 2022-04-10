# Matrix Multiplication with CUDA
Implement a way to compute matrix multiplication with CUDA by computing sub-matrices in parallel with multiple threads. The optimized kernel is then compared to the rudimentary way of implementing matrix multiplication with the CPU.

## Build
Navigate to the root directory of this project and build with CMake. My platform is Windows and I built it with 
```sh
   C:\msys64\mingw64\bin\cmake.EXE --build c:/Users/XXX/cuda_projects/matrix_mul/build --config Debug --target ALL_BUILD -j 10 --
```


## Results
For the test below, two $1^{10} \times 1^{10}$ square matrices have been multiplied together.
| Method | Time taken (ms) |
| --- | --- |
| Rudimentory matrix multiplication with CPU | 3785390 |
| Matrix multiplication with CUDA | 23 |