cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

if(BUILD_TESTS)
    message("")
    find_package(PythonInterp)
    # Set location of test map
    set(TEST_MAP_FILE "${PROJECT_BINARY_DIR}/test_map.npz")

    # Create test map
    execute_process(COMMAND ${PYTHON_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/scripts/autorally/test/generateTestMaps.py
                    -o ${TEST_MAP_FILE}
                    OUTPUT_FILE /dev/null
                    )
    message(STATUS "Created ${TEST_MAP_FILE}")

    # Create header file pointing to test map
    set(TEST_MAP_HEADER_FILE ${PROJECT_BINARY_DIR}/include/autorally_test_map.h)
    configure_file(autorally_test_map.h.in ${TEST_MAP_HEADER_FILE})
    message(STATUS "Generated ${TEST_MAP_HEADER_FILE}")
    # Add the generated header file to list of includes
    include_directories(${PROJECT_BINARY_DIR}/include)
    message("")
endif()

add_executable(ar_cost_tests
               ../test_main.cpp
               autorally_standard_cost_test.cu)

target_link_libraries(ar_cost_tests gtest
                      gmock
                      gtest_main
                      ar_standard_cost_kernel_test
                      cnpy)

add_test(NAME ar_cost_tests COMMAND ar_cost_tests)
set_target_properties(ar_cost_tests PROPERTIES FOLDER test)


add_executable(cp_quadratic_cost_tests
        ../test_main.cpp
        cartpole_quadratic_cost_test.cu)

target_link_libraries(cp_quadratic_cost_tests gtest
        gmock
        gtest_main
        cartpole_quadratic_cost_kernel_test
        cnpy)

add_test(NAME cp_quadratic_cost_tests COMMAND cp_quadratic_cost_tests)
set_target_properties(cp_quadratic_cost_tests PROPERTIES FOLDER test)
