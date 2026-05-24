cmake_minimum_required(VERSION 3.24)

project(
    cuda_playground
    VERSION 0.1
    LANGUAGES CXX CUDA
    )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file (GLOB EXAMPLE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/examples examples/*)
foreach(EXAMPLE_DIR ${EXAMPLE_DIRS})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/${EXAMPLE_DIR})
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/${EXAMPLE_DIR})
    endif()
endforeach()