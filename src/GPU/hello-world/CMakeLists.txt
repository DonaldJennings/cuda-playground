project(hello-world LANGUAGES CUDA)

get_filename_component(EXAMPLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${EXAMPLE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/hello-world.cu
)

set_target_properties(${EXAMPLE_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)