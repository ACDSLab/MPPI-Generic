cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)


set(GTEST_LIBRARIES gtest gmock gtest_main)
# Generic dynamics test
set(TARGET_NAME dynamics_generic_tests)

add_executable(${TARGET_NAME} ../test_main.cpp dynamics_generic_tests.cu)
target_link_libraries(${TARGET_NAME} ${GTEST_LIBRARIES})

add_test(NAME ${TARGET_NAME} COMMAND dynamics_generic_tests)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

# Cartpole dynamics tests
set(TARGET_NAME cartpole_dynamics_tests)

add_executable(${TARGET_NAME} ../test_main.cpp cartpole_dynamics_tests.cu)
target_link_libraries(${TARGET_NAME} ${GTEST_LIBRARIES})

add_test(NAME ${TARGET_NAME} COMMAND cartpole_dynamics_tests)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

if(BUILD_TESTS)
  message("")
  find_package(PythonInterp)
  # Set location of test map
  set(TEST_NETWORK_FOLDER "${PROJECT_BINARY_DIR}")

  # Create neural network map
  execute_process(COMMAND ${PYTHON_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/scripts/autorally/test/generateTestNetwork.py
          -o ${TEST_NETWORK_FOLDER}
          OUTPUT_FILE /dev/null
          )
  message(STATUS "Created ${TEST_NETWORK_FOLDER}/*.npz")

  # Create header file pointing to test neural network
  set(TEST_NETWORK_HEADER_FILE ${PROJECT_BINARY_DIR}/include/autorally_test_network.h)
  configure_file(autorally_test_network.h.in ${TEST_NETWORK_HEADER_FILE})
  message(STATUS "Generated ${TEST_NETWORK_HEADER_FILE}")
  # Add the generated header file to list of includes
  include_directories(${PROJECT_BINARY_DIR}/include)
  message("")
endif()

# Autorally Dynamics tests
set(AR_DYN_TARGET_NAME ar_nn_dynamics_tests)
add_executable(${AR_DYN_TARGET_NAME} ../test_main.cpp ar_dynamics_nn_test.cu)
target_link_libraries(${AR_DYN_TARGET_NAME}
                      ${GTEST_LIBRARIES}
                      cnpy)

add_test(NAME ${AR_DYN_TARGET_NAME} COMMAND ${AR_DYN_TARGET_NAME})
set_target_properties(${AR_DYN_TARGET_NAME} PROPERTIES FOLDER test)

## Autorally Dynamics Constant Memory Tests
add_executable(${AR_DYN_TARGET_NAME}_constant ../test_main.cpp ar_dynamics_nn_test.cu)
target_compile_definitions(${AR_DYN_TARGET_NAME}_constant PRIVATE MPPI_NNET_USING_CONSTANT_MEM__)
target_link_libraries(${AR_DYN_TARGET_NAME}_constant ${GTEST_LIBRARIES} cnpy)

add_test(NAME ${AR_DYN_TARGET_NAME}_constant COMMAND ${AR_DYN_TARGET_NAME}_constant)
set_target_properties(${AR_DYN_TARGET_NAME}_constant PROPERTIES FOLDER test)

# Quadrotor Dynamics tests
set(QUADROTOR_DYN_TARGET_NAME quadrotor_dyn_tests)
add_executable(${QUADROTOR_DYN_TARGET_NAME} ../test_main.cpp quadrotor_dynamics_tests.cu)
target_link_libraries(${QUADROTOR_DYN_TARGET_NAME} ${GTEST_LIBRARIES})

add_test(NAME ${QUADROTOR_DYN_TARGET_NAME} COMMAND ${QUADROTOR_DYN_TARGET_NAME})
set_target_properties(${QUADROTOR_DYN_TARGET_NAME} PROPERTIES FOLDER test)