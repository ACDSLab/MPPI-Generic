cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)


add_executable(vanilla_mppi_tests
               ../test_main.cpp
        vanilla_mppi_test.cu)

target_link_libraries(vanilla_mppi_tests gtest
                      gmock
                      gtest_main
                      cartpole_mppi
                      ${CUDA_LIBRARIES}
                      cnpy)
gtest_discover_tests(vanilla_mppi_tests)
set_target_properties(vanilla_mppi_tests PROPERTIES FOLDER test)

#add_executable(controller_generic_tests
#        ../test_main.cpp
#        controller_generic_tests.cu)
#
#target_link_libraries(controller_generic_tests gtest
#        gmock
#        gtest_main
#        ${CUDA_LIBRARIES}
#        ${OpenCV_LIBRARIES}
#        curand
#        cnpy)
#gtest_discover_tests(controller_generic_tests)
#set_target_properties(controller_generic_tests PROPERTIES FOLDER test)


add_executable(tube_mppi_tests
        ../test_main.cpp
        tube_mppi_test.cu)
target_link_libraries(tube_mppi_tests gtest
        gmock
        gtest_main
        curand
        ${CUDA_LIBRARIES}
        cnpy)
gtest_discover_tests(tube_mppi_tests)
set_target_properties(tube_mppi_tests PROPERTIES FOLDER test)

add_executable(ddp_tests
        ../test_main.cpp
        ddp_test.cu)
target_link_libraries(ddp_tests gtest
        gmock
        gtest_main
        cartpole_mppi
        cnpy)
gtest_discover_tests(ddp_tests)
set_target_properties(ddp_tests PROPERTIES FOLDER test)

add_executable(rmppi_tests
        ../test_main.cpp
        rmppi_test.cu)
target_link_libraries(rmppi_tests gtest
        gmock
        gtest_main
        curand
        ${CUDA_LIBRARIES}
        cnpy)
gtest_discover_tests(rmppi_tests)

set_target_properties(rmppi_tests PROPERTIES FOLDER test)
