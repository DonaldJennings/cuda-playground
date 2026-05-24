
cmake_minimum_required(VERSION 3.20)

project(RangeDopplerProfileGPU LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(RangeDopplerProfileGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/range-profile.cu
)
target_link_libraries(RangeDopplerProfileGPU
    PRIVATE
    CUDA::cufft
)

set_target_properties(RangeDopplerProfileGPU
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)