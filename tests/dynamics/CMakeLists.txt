cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)


add_executable(cartpole_dynamics_tests ../test_main.cpp cartpole_dynamics_tests.cpp)
target_link_libraries(cartpole_dynamics_tests gtest gmock gtest_main dynamics_kernel_tests)

add_test(NAME cartpole_dynamics_tests COMMAND cartpole_dynamics_tests)
set_target_properties(cartpole_dynamics_tests PROPERTIES FOLDER test)