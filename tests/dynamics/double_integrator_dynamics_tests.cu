//
// Created by mgandhi3 on 3/4/20.
//
#include <gtest/gtest.h>
#include <dynamics/double_integrator/di_dynamics.cuh>
#include "di_dynamics_kernel_tests.cuh"
#include <memory>

TEST(DI_Dynamics, CompareModelSize) {
  auto model = std::make_shared<DoubleIntegratorDynamics>();

  model->GPUSetup();

  long* model_size_GPU_d;

  HANDLE_ERROR(cudaMalloc((void**)&model_size_GPU_d, sizeof(long)));

  CheckModelSize<<<1,1>>>(model->model_d_, model_size_GPU_d);
  CudaCheckError();

  long model_size_GPU;
  HANDLE_ERROR(cudaMemcpy(&model_size_GPU, model_size_GPU_d, sizeof(long), cudaMemcpyDeviceToHost));

  ASSERT_EQ(sizeof(*model), model_size_GPU);

  std::cout << "Size of the shared pointer to the model:" << sizeof(model) << std::endl; // Should be the size of a pointer so 8 bytes for a 64 bit system

  std::cout << "Size of the model itself: " << sizeof(*model) << std::endl; // Should be bigger?

  std::cout << "Size of the model on the GPU: " << model_size_GPU << std::endl;

  std::cout << "Size of the control ranges in the model: " << sizeof(model->control_rngs_) << std::endl; // Should be 16 bytes ie. 4 floats!

  std::cout << "Size of the parameter structure of the model: " << sizeof(DoubleIntegratorParams) << std::endl;

  std::cout << "Size of the device pointer of the model: " << sizeof(model->model_d_) << std::endl;

  std::cout << "Size of the stream: " << sizeof(model->stream_) << std::endl;

  std::cout << "Size of GPU Memstatus: " << sizeof(model->GPUMemStatus_) << std::endl;

}