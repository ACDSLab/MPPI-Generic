set(GTEST_LIBRARIES gtest gmock gtest_main)
file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  if(${T_NAME} MATCHES buffered_plant_tester)
    if (EXISTS /usr/bin/gfortran)
      add_executable(${T_NAME} ../test_main.cpp ${T_FILE})
      target_link_libraries(${T_NAME}
              ${GTEST_LIBRARIES}
              ${OpenCV_LIBRARIES}
              ${MPPI_GENERIC_CUDA_EXTRA_LIBS}
              cnpy)
      gtest_add_tests(TARGET ${T_NAME})
      set_target_properties(${T_NAME} PROPERTIES FOLDER test)
    endif()
  else()
    add_executable(${T_NAME} ../test_main.cpp ${T_FILE})
    target_link_libraries(${T_NAME}
              ${GTEST_LIBRARIES}
              ${OpenCV_LIBRARIES}
              ${MPPI_GENERIC_CUDA_EXTRA_LIBS}
              cnpy)
      gtest_add_tests(TARGET ${T_NAME})
    set_target_properties(${T_NAME} PROPERTIES FOLDER test)
    endif()
endforeach()
