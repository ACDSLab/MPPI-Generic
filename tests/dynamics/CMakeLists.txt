if(BUILD_TESTS)
  message("")
  find_package(PythonInterp)

  # Create neural network map
  execute_process(COMMAND ${PYTHON_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/scripts/autorally/test/generateTestNetwork.py
          -o ${TEST_NETWORK_FOLDER}
          OUTPUT_FILE /dev/null
          )
  message(STATUS "Created ${TEST_NETWORK_FOLDER}/*.npz")
  message("")
endif()

set(GTEST_LIBRARIES gtest gmock gtest_main)
file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  add_executable(${T_NAME} ../test_main.cpp ${T_FILE})
  target_link_libraries(${T_NAME}
                        ${GTEST_LIBRARIES}
                        cnpy
                        ${CUDA_curand_LIBRARY})
  gtest_discover_tests(${T_NAME})
  set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()

## Autorally Dynamics Constant Memory Tests
#add_executable(ar_dynamics_nn_test_constant ../test_main.cpp ar_dynamics_nn_test.cu)
#target_compile_definitions(ar_dynamics_nn_test_constant PRIVATE MPPI_NNET_USING_CONSTANT_MEM__)
#target_link_libraries(ar_dynamics_nn_test_constant
#                      ${GTEST_LIBRARIES}
#                      cnpy)
#
#gtest_add_tests(TARGET ar_dynamics_nn_test_constant)
#
#set_target_properties(ar_dynamics_nn_test_constant PROPERTIES FOLDER test)
