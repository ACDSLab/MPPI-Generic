add_executable(cartpole_example cartpole_example.cu)
target_link_libraries(cartpole_example
                      ${CUDA_curand_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      cnpy
                      cartpole_mppi)
