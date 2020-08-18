add_executable(cartpole_example cartpole_example.cpp)
target_include_directories(cartpole_example PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cartpole_example
                      ${OpenCV_LIBRARIES}
                      cnpy
                      cartpole_mppi)

add_executable(double_integrator_CORL2020 double_integrator_CORL2020.cu)
target_include_directories(double_integrator_CORL2020 PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(double_integrator_CORL2020
        curand
        ${CUDA_LIBRARIES}
        cnpy)
