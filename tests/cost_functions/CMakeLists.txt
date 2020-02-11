cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

if(BUILD_TESTS)
    message("")
    find_package(PythonInterp)
    # Set location of test map
    set(TEST_MAP_FOLDER "${PROJECT_BINARY_DIR}")

    # Create test map
    execute_process(COMMAND ${PYTHON_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/scripts/autorally/test/generateTestMaps.py
                    -o ${TEST_MAP_FOLDER}/
                    OUTPUT_FILE /dev/null
                    )
    message(STATUS "Created folder ${TEST_MAP_FOLDER}/*.npz")

    # Create header file pointing to test map
    set(TEST_MAP_HEADER_FILE ${PROJECT_BINARY_DIR}/include/autorally_test_map.h)
    configure_file(autorally_test_map.h.in ${TEST_MAP_HEADER_FILE})
    message(STATUS "Generated ${TEST_MAP_HEADER_FILE}")
    # Add the generated header file to list of includes
    include_directories(${PROJECT_BINARY_DIR}/include)
    message("")
endif()

set(TEST_NAME ar_cost_tests)

add_executable(${TEST_NAME}
               ../test_main.cpp
               autorally_standard_cost_test.cu)

target_link_libraries(${TEST_NAME} gtest
                      gmock
                      gtest_main
                      #ar_standard_cost_kernel_test
                      cnpy)

add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
set_target_properties(${TEST_NAME} PROPERTIES FOLDER test)


set(TEST_NAME ar_robust_cost_tests)

add_executable(${TEST_NAME}
        ../test_main.cpp
        autorally_robust_cost_test.cu)

target_link_libraries(${TEST_NAME} gtest
        gmock
        gtest_main
        #ar_standard_cost_kernel_test
        #ar_robust_cost_kernel_test
        cnpy)

add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
set_target_properties(${TEST_NAME} PROPERTIES FOLDER test)
