cmake_minimum_required(VERSION 3.0.0)
project(matrixmul VERSION 2.0.0 LANGUAGES CXX CUDA)

enable_language(CUDA)

# Configure file to allow access CMake variables from your code
configure_file(
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h"
)

add_executable(${PROJECT_NAME} main.cu)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR})

add_subdirectory(matrixmul_kernel)
add_subdirectory(cublas_kernel)

target_link_libraries(${PROJECT_NAME} matrixmul_kernel cublas_kernel)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set_target_properties(
        matrixmul
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
include(CPack)
