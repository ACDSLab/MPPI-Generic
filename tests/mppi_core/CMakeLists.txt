# Rollout kernel test suite
set(ROLLOUT_TARGET_NAME rollout_kernel_tests)

add_executable(${ROLLOUT_TARGET_NAME} ../test_main.cpp rollout_kernel_tests.cu)
target_link_libraries(${ROLLOUT_TARGET_NAME} gtest
        gmock
        gtest_main
        curand
        cnpy)

gtest_discover_tests(${ROLLOUT_TARGET_NAME})
set_target_properties(${ROLLOUT_TARGET_NAME} PROPERTIES FOLDER test)

# Norm exponential kernel test suite
set(NORMEXP_TARGET_NAME normexp_kernel_tests)

add_executable(${NORMEXP_TARGET_NAME} ../test_main.cpp normexp_kernel_tests.cu)
target_link_libraries(${NORMEXP_TARGET_NAME} gtest
        gmock
        gtest_main
        curand)

gtest_discover_tests(${NORMEXP_TARGET_NAME})
set_target_properties(${NORMEXP_TARGET_NAME} PROPERTIES FOLDER test)

# Weighted Reduction Kernel Test Suite
set(WEIGHTRED_TARGET_NAME weightedreduction_kernel_tests)

add_executable(${WEIGHTRED_TARGET_NAME} ../test_main.cpp
               weightedreduction_kernel_tests.cu)
target_link_libraries(${WEIGHTRED_TARGET_NAME} gtest
        gmock
        gtest_main
        curand)

gtest_discover_tests(${WEIGHTRED_TARGET_NAME})
set_target_properties(${WEIGHTRED_TARGET_NAME} PROPERTIES FOLDER test)

# Base Plant test cases
set(PLANT_TARGET_NAME base_plant_tests)

add_executable(${PLANT_TARGET_NAME} ../test_main.cpp base_plant_tester.cu)
target_link_libraries(${PLANT_TARGET_NAME} gtest
        gmock
        gtest_main
        ${OpenCV_LIBRARIES}
        )

gtest_discover_tests(${PLANT_TARGET_NAME})
set_target_properties(${PLANT_TARGET_NAME} PROPERTIES FOLDER test)

set(PLANT_TARGET_NAME buffered_plant_tests)

add_executable(${PLANT_TARGET_NAME} ../test_main.cpp buffered_plant_tester.cu)
target_link_libraries(${PLANT_TARGET_NAME} gtest
        gmock
        gtest_main
        fitpackpp
        ${OpenCV_LIBRARIES}
        )

gtest_discover_tests(${PLANT_TARGET_NAME})
set_target_properties(${PLANT_TARGET_NAME} PROPERTIES FOLDER test)

# RMPPI Kernel Test Cases
set(RMPPIKERNEL_TARGET_NAME rmppi_kernel_tests)

add_executable(${RMPPIKERNEL_TARGET_NAME} ../test_main.cpp
        rmppi_kernel_tests.cu)
target_link_libraries(${RMPPIKERNEL_TARGET_NAME} gtest
        gmock
        gtest_main
        curand)

gtest_discover_tests(${RMPPIKERNEL_TARGET_NAME})
set_target_properties(${RMPPIKERNEL_TARGET_NAME} PROPERTIES FOLDER test)

# RMPPI Kernel Test Cases
set(CCM_TARGET_NAME CCM_tests)

add_executable(${CCM_TARGET_NAME} ../test_main.cpp
        CCM_tests.cu)
target_link_libraries(${CCM_TARGET_NAME} gtest
        gmock
        gtest_main
        curand
        cnpy)

gtest_discover_tests(${CCM_TARGET_NAME})
set_target_properties(${CCM_TARGET_NAME} PROPERTIES FOLDER test)
