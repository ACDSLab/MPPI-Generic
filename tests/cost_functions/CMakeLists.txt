cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

add_executable(ar_cost_tests ../test_main.cpp autorally_standard_cost_test.cu)
# TODO not generic
#target_link_libraries(ar_cost_tests gtest gmock gtest_main ar_standard_cost_kernel_test /usr/local/lib/libcnpy.so)
target_link_libraries(ar_cost_tests gtest gmock gtest_main ar_standard_cost_kernel_test /home/jgibson37/.local/lib/libcnpy.so)

add_test(NAME ar_cost_tests COMMAND ar_cost_tests)
set_target_properties(ar_cost_tests PROPERTIES FOLDER test)
