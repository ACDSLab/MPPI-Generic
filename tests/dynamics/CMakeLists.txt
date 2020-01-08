cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME cartpole_dynamics_tests)

add_executable(${TARGET_NAME} ../test_main.cpp cartpole_dynamics_tests.cpp)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main dynamics_kernel_tests)

add_test(NAME ${TARGET_NAME} COMMAND cartpole_dynamics_tests)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)