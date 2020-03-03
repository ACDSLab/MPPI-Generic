cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)
# Rollout kernel test suite
set(ROLLOUT_TARGET_NAME rollout_kernel_tests)

add_executable(${ROLLOUT_TARGET_NAME} ../test_main.cpp rollout_kernel_tests.cu)
target_link_libraries(${ROLLOUT_TARGET_NAME} gtest
        gmock
        gtest_main
        curand)

add_test(NAME ${ROLLOUT_TARGET_NAME} COMMAND rollout_kernel_tests)
set_target_properties(${ROLLOUT_TARGET_NAME} PROPERTIES FOLDER test)

# Norm exponential kernel test suite
set(NORMEXP_TARGET_NAME normexp_kernel_tests)

add_executable(${NORMEXP_TARGET_NAME} ../test_main.cpp normexp_kernel_tests.cu)
target_link_libraries(${NORMEXP_TARGET_NAME} gtest
        gmock
        gtest_main
        curand)

add_test(NAME ${NORMEXP_TARGET_NAME} COMMAND normexp_kernel_tests)
set_target_properties(${NORMEXP_TARGET_NAME} PROPERTIES FOLDER test)

# Weighted Reduction Kernel Test Suite
set(WEIGHTRED_TARGET_NAME weightedreduction_kernel_tests)

add_executable(${WEIGHTRED_TARGET_NAME} ../test_main.cpp weightedreduction_kernel_tests.cu)
target_link_libraries(${WEIGHTRED_TARGET_NAME} gtest
        gmock
        gtest_main
        curand)

add_test(NAME ${WEIGHTRED_TARGET_NAME} COMMAND weightedreduction_kernel_tests)
set_target_properties(${WEIGHTRED_TARGET_NAME} PROPERTIES FOLDER test)

# Base Plant test cases
set(PLANT_TARGET_NAME base_plant_tests)

add_executable(${PLANT_TARGET_NAME} ../test_main.cpp base_plant_tester.cpp)
target_link_libraries(${PLANT_TARGET_NAME} gtest
        gmock
        gtest_main)

add_test(NAME ${PLANT_TARGET_NAME} COMMAND base_plant_tests)
set_target_properties(${PLANT_TARGET_NAME} PROPERTIES FOLDER test)
