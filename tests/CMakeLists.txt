# Add project includes and cuda includes for g++
include_directories(${PROJECT_SOURCE_DIR}/kernel_tests/
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#include test utils
include_directories(include)

# Create header file pointing to test neural network
set(HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/templated_headers/)

file(GLOB HEADERS ${HEADER_PATH}*)
foreach(T_FILE IN LISTS HEADERS)
    get_filename_component(T_NAME ${T_FILE} NAME_WE)
    string(REPLACE ".in" "" STRIPPED_NAME ${T_NAME})
    message("stripped name ${STRIPPED_NAME}")
    set(TEST_NETWORK_HEADER_FILE ${PROJECT_BINARY_DIR}/include/${STRIPPED_NAME}.h)
    configure_file(${T_FILE} ${TEST_NETWORK_HEADER_FILE})
    message(STATUS "Generated ${TEST_NETWORK_HEADER_FILE}")
endforeach()

# Create header file pointing to test map
set(TEST_MAP_FOLDER "${PROJECT_BINARY_DIR}")

# Add the generated header files to list of includes
include_directories(${PROJECT_BINARY_DIR}/include)
message("")

add_subdirectory(dynamics)
add_subdirectory(cost_functions)
add_subdirectory(feedback_controllers)
add_subdirectory(mppi_core)
add_subdirectory(controllers)
add_subdirectory(misc)
add_subdirectory(sampling_distributions)
add_subdirectory(shaping_functions)
add_subdirectory(math_utils)
add_subdirectory(integration)
add_subdirectory(texture_helpers)
add_subdirectory(nn_helpers)
