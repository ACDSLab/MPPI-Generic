cmake_minimum_required(VERSION 3.00)
project(MPPIGeneric)

set(BUILD_FLAGS "-std=c++11 -Wuninitialized -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_FLAGS}")

# Allow the lib location to be overwritten from command line
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    get_filename_component(PROJECT_LIBS_DIR ${PROJECT_BINARY_DIR}/lib ABSOLUTE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
    message("${PROJECT_LIBS_DIR}")
else()
    get_filename_component(PROJECT_LIBS_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ABSOLUTE)
endif()

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# set up cnpy
include_directories(submodules/cnpy)

find_package(CUDA REQUIRED)

if(NOT ${CUDA_curand_LIBRARY} MATCHES "NOTFOUND")
    set(CURAND_LIBRARY ${CUDA_curand_LIBRARY})
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -std=c++11
        -maxrregcount=32;
        -O3;
        -arch=sm_35;
        -w
)

add_subdirectory(MPPI-Generic)
