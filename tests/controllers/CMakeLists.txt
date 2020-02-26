cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)


add_executable(vanilla_mppi_tests
               ../test_main.cpp
               vanilla_mppi_test.cpp)

target_link_libraries(vanilla_mppi_tests gtest
                      gmock
                      gtest_main
                      cartpole_mppi
                      ${CUDA_LIBRARIES}
                      cnpy)
add_test(NAME vanilla_mppi_tests COMMAND vanilla_mppi_tests)
set_target_properties(vanilla_mppi_tests PROPERTIES FOLDER test)


add_executable(tube_mppi_tests
        ../test_main.cpp
        tube_mppi_test.cpp)
target_link_libraries(tube_mppi_tests gtest
        gmock
        gtest_main
        double_integrator_mppi
        cnpy
        ${Boost_LIBRARIES})
add_test(NAME tube_mppi_tests COMMAND tube_mppi_tests)
set_target_properties(tube_mppi_tests PROPERTIES FOLDER test)

add_executable(ddp_tests
        ../test_main.cpp
        ddp_test.cu)
target_link_libraries(ddp_tests gtest
        gmock
        gtest_main
        cartpole_mppi
        cnpy
        ${Boost_LIBRARIES})
add_test(NAME ddp_tests COMMAND ddp_tests)
set_target_properties(ddp_tests PROPERTIES FOLDER test)