cmake_minimum_required(VERSION 3.24)

project(
    cuda_playground
    VERSION 0.1
    LANGUAGES CXX
    )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit QUIET)

if (NOT CUDAToolkit_FOUND)
    set(HAS_CUDA_TOOLKIT FALSE)
    message(STATUS "CUDA Toolkit not found. Please install CUDA and set the appropriate environment variables. Building without CUDA support.")
else()
    set(HAS_CUDA_TOOLKIT TRUE)
    message(STATUS "CUDA Toolkit bfound: ${CUDAToolkit_VERSION}")
endif()

file(GLOB CPU_EXAMPLE_DIRS "src/CPU/*")
file(GLOB GPU_EXAMPLE_DIRS "src/GPU/*")

# Iterate over each subdirectory in CPU/* and call add_subdirectory
foreach(CPU_DIR ${CPU_EXAMPLE_DIRS})
    if (IS_DIRECTORY ${CPU_DIR})
        add_subdirectory(${CPU_DIR})
    endif()
endforeach()

if (HAS_CUDA_TOOLKIT)
    # Iterate over each subdirectory in GPU/* and call add_subdirectory
    foreach(GPU_DIR ${GPU_EXAMPLE_DIRS})
        if (IS_DIRECTORY ${GPU_DIR})
            add_subdirectory(${GPU_DIR})
        endif()
    endforeach()
endif()