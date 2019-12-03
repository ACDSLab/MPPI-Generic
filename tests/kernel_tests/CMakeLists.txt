cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)



add_executable(rollout_kernel_tests ../test_main.cpp rollout_kernel_tests.cu)
target_include_directories(rollout_kernel_tests PUBLIC ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/mppi_core)
target_link_libraries(rollout_kernel_tests gtest gmock gtest_main)


add_test(NAME rollout_kernel_tests COMMAND rollout_kernel_tests)
set_target_properties(rollout_kernel_tests PROPERTIES FOLDER test)