cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME rollout_kernel_tests)


add_executable(${TARGET_NAME} ../test_main.cpp rollout_kernel_tests.cpp)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main mppi_core_kernel_tests dynamics_kernel_tests)


add_test(NAME ${TARGET_NAME} COMMAND rollout_kernel_tests)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)