cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)


add_executable(math_utils_tests
  ../test_main.cpp
  math_utils_test.cu)

target_link_libraries(math_utils_tests gtest
  gmock
  gtest_main
  cnpy)
gtest_discover_tests(math_utils_tests)
set_target_properties(math_utils_tests PROPERTIES FOLDER test)
