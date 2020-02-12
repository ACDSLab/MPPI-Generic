cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME cartpole_dynamics_tests)

add_executable(${TARGET_NAME} ../test_main.cpp cartpole_dynamics_tests.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main cartpole_dynamics_kernel_tests)

add_test(NAME ${TARGET_NAME} COMMAND cartpole_dynamics_tests)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

if(BUILD_TESTS)
  message("")
  find_package(PythonInterp)
  # Set location of test map
  set(TEST_NETWORK_FILE "${PROJECT_BINARY_DIR}/test_network.npz")

  # Create neural network map
  execute_process(COMMAND ${PYTHON_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/scripts/autorally/test/generateTestNetwork.py
          -o ${TEST_NETWORK_FILE}
          OUTPUT_FILE /dev/null
          )
  message(STATUS "Created ${TEST_NETWORK_FILE}")

  # Create header file pointing to test neural network
  set(TEST_NETWORK_HEADER_FILE ${PROJECT_BINARY_DIR}/include/autorally_test_network.h)
  configure_file(autorally_test_network.h.in ${TEST_NETWORK_HEADER_FILE})
  message(STATUS "Generated ${TEST_NETWORK_HEADER_FILE}")
  # Add the generated header file to list of includes
  include_directories(${PROJECT_BINARY_DIR}/include)
  message("")
endif()

add_executable(ar_nn_dynamics_tests ../test_main.cpp ar_dynamics_nn_test.cu)
target_link_libraries(ar_nn_dynamics_tests gtest gmock gtest_main ar_nn_dynamics_kernel_test cnpy)

add_test(NAME ar_nn_dynamics_tests COMMAND ar_nn_dynamics_tests)
set_target_properties(ar_nn_dynamics_tests PROPERTIES FOLDER test)

add_executable(ar_nn_dynamics_tests_constant ../test_main.cpp ar_dynamics_nn_test.cu)
target_compile_definitions(ar_nn_dynamics_tests_constant PRIVATE MPPI_NNET_USING_CONSTANT_MEM__)
target_link_libraries(ar_nn_dynamics_tests_constant gtest gmock gtest_main ar_nn_dynamics_kernel_test cnpy)

add_test(NAME ar_nn_dynamics_tests_constant COMMAND ar_nn_dynamics_tests)
set_target_properties(ar_nn_dynamics_tests_constant PROPERTIES FOLDER test)
