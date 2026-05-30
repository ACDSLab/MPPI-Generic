set(GTEST_LIBRARIES gtest gmock gtest_main)
file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

# HIP port: the texture_helper tests exercise the deferred texture-backed map-cost
# sub-feature (cudaFilterModeLinear + element-read float, rejected on AMD; needs
# software bilinear/trilinear interp). All three texture_helper_test TUs are
# deferred, so exclude them on the HIP build to keep a default build-all green.
# CUDA build is unchanged.
if (USE_HIP)
  list(REMOVE_ITEM TARGET_SRCS
    texture_helper_test.cu
    two_d_texture_helper_test.cu
    three_d_texture_helper_test.cu
  )
endif()

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  add_executable(${T_NAME} ${PROJECT_SOURCE_DIR}/tests/test_main.cpp ${T_FILE})
  target_link_libraries(${T_NAME}
          ${GTEST_LIBRARIES}
          ${MPPI_HEADER_LIBRARY_NAME})
  gtest_discover_tests(${T_NAME})
  set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()
