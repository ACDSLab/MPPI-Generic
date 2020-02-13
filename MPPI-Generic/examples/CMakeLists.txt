add_executable(cartpole_example cartpole_example.cpp)
target_include_directories(cartpole_example PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cartpole_example
                      ${CUDA_curand_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      cnpy
                      cartpole_mppi)
