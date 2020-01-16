cmake_minimum_required(VERSION 3.8)
project(MPPIGeneric LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(BUILD_FLAGS "-Wuninitialized -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_FLAGS}")

option(BUILD_TESTS "BUILD_TESTS" ON)

# Allow the lib location to be overwritten from command line
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    get_filename_component(PROJECT_LIBS_DIR ${PROJECT_BINARY_DIR}/lib ABSOLUTE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
else()
    get_filename_component(PROJECT_LIBS_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ABSOLUTE)
endif()

# Find Eigen
find_package(Eigen3 REQUIRED)
# TODO not generic
#include_directories(/home/jason/.local/include/eigen3)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen loc: ${EIGEN3_INCLUDE_DIR}")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

# find yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# set up cnpy
# Don't install cnpy to user package registry
set(TMP_PACKAGE_REGISTRY_EXPORT ${CMAKE_EXPORT_NO_PACKAGE_REGISTRY})
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
add_subdirectory(submodules/cnpy)
include_directories(submodules/cnpy)
# Set the no package registry back to default
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ${TMP_PACKAGE_REGISTRY_EXPORT})

#find_package(CUDA REQUIRED)

if(NOT ${CUDA_curand_LIBRARY} MATCHES "NOTFOUND")
    set(CURAND_LIBRARY ${CUDA_curand_LIBRARY})
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -maxrregcount=32;
        -arch=sm_35;
        -w;
)

add_subdirectory(MPPI-Generic)

# Add custom cmake finds
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Use CMake to download gtest as part of the configure step
###################################################################
# Add gtest
###################################################################
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE AND BUILD_TESTS)
  enable_testing()

  ############################################################
  # copied from
  # https://github.com/google/googletest/tree/master/googletest#incorporating-into-an-existing-cmake-project
  ############################################################

  # Download and unpack googletest at configure time
  if (NOT TARGET gtest_main)
    configure_file(${CMAKE_MODULE_PATH}/CMakeLists.txt.gtest.in
	${PROJECT_BINARY_DIR}/googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${PROJECT_BINARY_DIR}/googletest-src
                     ${PROJECT_BINARY_DIR}/googletest-build)
  endif()

  add_subdirectory(tests)
endif()
