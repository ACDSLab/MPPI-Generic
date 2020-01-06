# Add project includes and cuda includes for g++
include_directories(${CMAKE_SOURCE_DIR}/MPPI-Generic/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(dynamics_tests)
add_subdirectory(kernel_tests)
add_subdirectory(cuda_stream_tests)