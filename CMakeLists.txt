cmake_minimum_required(VERSION 3.5.0)
project(matrixmul VERSION 2.0.0 LANGUAGES CXX CUDA)

enable_language(CUDA)

# Configure file to allow access CMake variables from your code
configure_file(
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h"
)

# Download all submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/cxxopts/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

add_subdirectory(cxxopts)
add_subdirectory(matrixmul_kernel)
add_subdirectory(cublas_kernel)

add_executable(${PROJECT_NAME} main.cu)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR} PUBLIC cxxopts/include)

target_link_directories(${PROJECT_NAME}
        PRIVATE matrixmul_kernel
        PRIVATE cublas_kernel
        PRIVATE cxxopts/src)

target_link_libraries(${PROJECT_NAME} 
        matrixmul_kernel
        cublas_kernel
        cxxopts)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set_target_properties(
        matrixmul
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
include(CPack)
