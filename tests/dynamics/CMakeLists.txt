cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME cartpole_dynamics_tests)

add_executable(${TARGET_NAME} ../test_main.cpp cartpole_dynamics_tests.cpp)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main dynamics_kernel_tests)

add_test(NAME ${TARGET_NAME} COMMAND cartpole_dynamics_tests)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

add_executable(ar_nn_dynamics_tests ../test_main.cpp ar_dynamics_nn_test.cu)
target_link_libraries(ar_nn_dynamics_tests gtest gmock gtest_main ar_nn_dynamics_kernel_test cnpy)

add_test(NAME ar_nn_dynamics_tests COMMAND ar_nn_dynamics_tests)
set_target_properties(ar_nn_dynamics_tests PROPERTIES FOLDER test)
