# Rollout kernel test suite
set(SHAPING_FUNCTION_TARGET_NAME shaping_function_tests)

add_executable(${SHAPING_FUNCTION_TARGET_NAME} ../test_main.cpp shaping_function_test.cu)
target_link_libraries(${SHAPING_FUNCTION_TARGET_NAME} gtest
        gmock
        gtest_main
        curand
        cnpy)

gtest_discover_tests(${SHAPING_FUNCTION_TARGET_NAME})
set_target_properties(${SHAPING_FUNCTION_TARGET_NAME} PROPERTIES FOLDER test)
