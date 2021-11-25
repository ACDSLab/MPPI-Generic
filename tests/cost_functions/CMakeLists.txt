if(BUILD_TESTS)
    message("")
    find_package(PythonInterp)

    # Create test map
    execute_process(COMMAND ${PYTHON_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/scripts/autorally/test/generateTestMaps.py
                    -o ${TEST_MAP_FOLDER}/
                    OUTPUT_FILE /dev/null
                    )
    message(STATUS "Created folder ${TEST_MAP_FOLDER}/*.npz")
    message("")
endif()

set(GTEST_LIBRARIES gtest gmock gtest_main)
file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  add_executable(${T_NAME} ../test_main.cpp ${T_FILE})
  target_link_libraries(${T_NAME}
                        ${GTEST_LIBRARIES}
                        cnpy)
#  add_test(NAME ${T_NAME} COMMAND ${T_NAME})
  gtest_add_tests(TARGET ${T_NAME})
  set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()
