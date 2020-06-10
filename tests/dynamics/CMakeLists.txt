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

set(GTEST_LIBRARIES gtest gmock gtest_main)
# set(TARGETS ar_dynamics_nn_test quadrotor_dynamics_tests
#     cartpole_dynamics_tests dynamics_generic_tests)

file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  add_executable(${T_NAME} ../test_main.cpp ${T_FILE})
  target_link_libraries(${T_NAME}
                        ${GTEST_LIBRARIES}
                        cnpy)
  add_test(NAME ${T_NAME} COMMAND ${T_NAME})
  set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()

## Autorally Dynamics Constant Memory Tests
add_executable(ar_dynamics_nn_test_constant ../test_main.cpp ar_dynamics_nn_test.cu)
target_compile_definitions(ar_dynamics_nn_test_constant PRIVATE MPPI_NNET_USING_CONSTANT_MEM__)
target_link_libraries(ar_dynamics_nn_test_constant
                      ${GTEST_LIBRARIES}
                      cnpy)

add_test(NAME ar_dynamics_nn_test_constant COMMAND ar_dynamics_nn_test_constant)
set_target_properties(ar_dynamics_nn_test_constant PROPERTIES FOLDER test)
