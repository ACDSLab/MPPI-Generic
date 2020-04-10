# Add project includes and cuda includes for g++
include_directories(${PROJECT_SOURCE_DIR}/kernel_tests/
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(dynamics)
add_subdirectory(cost_functions)
add_subdirectory(mppi_core)
add_subdirectory(controllers)
add_subdirectory(misc)
