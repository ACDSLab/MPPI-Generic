cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)



add_executable(cuda_stream_tests ../test_main.cpp cuda_stream_tests.cpp)
target_link_libraries(cuda_stream_tests gtest gmock gtest_main)


add_test(NAME cuda_stream_tests COMMAND cuda_stream_tests)
set_target_properties(cuda_stream_tests PROPERTIES FOLDER test)