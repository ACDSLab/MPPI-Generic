# Add project includes and cuda includes for g++
include_directories(${PROJECT_SOURCE_DIR}/kernel_tests/
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#include test utils
include_directories(include)

# Create header file pointing to test neural network
set(TEST_NETWORK_FOLDER "${PROJECT_BINARY_DIR}")
set(TEST_NETWORK_HEADER_FILE ${PROJECT_BINARY_DIR}/include/autorally_test_network.h)
configure_file(templated_headers/autorally_test_network.h.in ${TEST_NETWORK_HEADER_FILE})
message(STATUS "Generated ${TEST_NETWORK_HEADER_FILE}")
# Create header file pointing to test map
set(TEST_MAP_FOLDER "${PROJECT_BINARY_DIR}")
set(TEST_MAP_HEADER_FILE ${PROJECT_BINARY_DIR}/include/autorally_test_map.h)
configure_file(templated_headers/autorally_test_map.h.in ${TEST_MAP_HEADER_FILE})
message(STATUS "Generated ${TEST_MAP_HEADER_FILE}")

# Add the generated header files to list of includes
include_directories(${PROJECT_BINARY_DIR}/include)
message("")

add_subdirectory(dynamics)
add_subdirectory(cost_functions)
add_subdirectory(mppi_core)
add_subdirectory(controllers)
add_subdirectory(misc)
