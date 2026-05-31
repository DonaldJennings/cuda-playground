cmake_minimum_required(VERSION 3.20)

project(RangeDopplerProfileGPU LANGUAGES CUDA CXX)

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
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# -lineinfo embeds source line information into PTX/SASS for Nsight profiling.
# It has negligible runtime cost and is invaluable for performance analysis.
target_compile_options(RangeDopplerProfileGPU
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
