cmake_minimum_required(VERSION 3.0.0)
project(matrixmul VERSION 0.1.0 LANGUAGES CXX CUDA)

enable_language(CUDA)
add_executable(${PROJECT_NAME} main.cu)
add_subdirectory(maltrixmul_kernel)

target_link_libraries(${PROJECT_NAME} maltrixmul_kernel)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set_target_properties(
        matrixmul
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
include(CPack)
