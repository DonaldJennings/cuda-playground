cmake_minimum_required(VERSION 3.24)

project(cuda_playground LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_playground
    src/main.cu
    src/kernels/vector_add.cu
)

target_include_directories(cuda_playground PRIVATE include)

set_target_properties(cuda_playground PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)