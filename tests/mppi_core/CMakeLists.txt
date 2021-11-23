set(GTEST_LIBRARIES gtest gmock gtest_main)
file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  add_executable(${T_NAME} ../test_main.cpp ${T_FILE})
  message("TEST: |${T_NAME}|")
  if(${T_NAME} MATCHES buffered_plant_tester)
    message("EQUAL")
    target_link_libraries(${T_NAME}
            ${GTEST_LIBRARIES}
            ${OpenCV_LIBRARIES}
            fitpackpp
            curand
            cnpy)
    gtest_add_tests(TARGET ${T_NAME})
  else()
    message("NOT EQUAL ${T_NAME}")
    target_link_libraries(${T_NAME}
              ${GTEST_LIBRARIES}
              ${OpenCV_LIBRARIES}
              curand
              cnpy)
      gtest_add_tests(TARGET ${T_NAME})
    endif()
  set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()
