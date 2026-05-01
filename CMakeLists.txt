cmake_minimum_required(VERSION 3.24)

project(
    cuda_playground
    VERSION 0.1
    LANGUAGES CXX CUDA
    )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cuda_playground
    src/main.cu
    src/kernels/vector_add.cu
)

target_include_directories(cuda_playground PRIVATE include)

target_compile_options(cuda_playground PRIVATE
    $<$<COMPILE_LANGUAGE::CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-G>
)

set_target_properties(cuda_playground PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)