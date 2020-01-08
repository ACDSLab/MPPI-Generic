cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

add_executable(cost_tests ../test_main.cpp autorally_standard_cost_test.cu)
target_link_libraries(cost_tests gtest gmock gtest_main ar_standard_cost)

add_test(NAME cost_tests COMMAND cost_tests)
set_target_properties(cost_tests PROPERTIES FOLDER test)
