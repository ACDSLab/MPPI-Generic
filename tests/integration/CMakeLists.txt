cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

add_executable(numerical_integration_test
        ../test_main.cpp
        integrator_tests.cu)

target_link_libraries(numerical_integration_test gtest
        gmock
        gtest_main
        cnpy)
gtest_discover_tests(numerical_integration_test)
set_target_properties(numerical_integration_test PROPERTIES FOLDER test)


