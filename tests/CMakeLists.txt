cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)



add_executable(cartpole_dynamics_tests test_main.cpp cartpole_dynamics_tests.cu)
target_include_directories(cartpole_dynamics_tests PUBLIC ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/dynamics)
target_link_libraries(cartpole_dynamics_tests cartpole_dynamics gtest gmock gtest_main)


add_test(NAME cartpole_dynamics_tests COMMAND cartpole_dynamics_tests)
set_target_properties(cartpole_dynamics_tests PROPERTIES FOLDER test)