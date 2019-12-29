# Add project includes and cuda includes for g++
include_directories(${CMAKE_SOURCE_DIR}/MPPI-Generic/include/controllers
                    ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/cost_functions
                    ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/dynamics
                    ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/mppi_core
                    ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/utils
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(dynamics_tests)
add_subdirectory(kernel_tests)
add_subdirectory(cuda_stream_tests)