#include <gtest/gtest.h>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/utils/test_helper.h>

#include <array>

TEST(DoubleIntegratorCost, Constructor)
{
  DoubleIntegratorCircleCost();
}

template <class COST_T>
void __global__ ControlCostKernel(COST_T* costs, float* u_d, float* eps_d, float* std_dev_d, float lambda,
                                  float* result)
{
  // Create u as on the GPU
  float u_noise[COST_T::CONTROL_DIM];
  for (int i = 0; i < COST_T::CONTROL_DIM; i++)
  {
    u_noise[i] = u_d[i] + eps_d[i];
  }
  result[0] = costs->computeLikelihoodRatioCost(u_noise, eps_d, std_dev_d, lambda);
  result[1] = costs->computeFeedbackCost(u_d, std_dev_d, lambda);
}

TEST(DoubleIntegratorCost, ControlCost)
{
  using COST = DoubleIntegratorCircleCost;
  const int num_results = 2;

  /**
   * CPU Setup
   */
  COST cost;
  COST::control_array u = COST::control_array::Ones();
  COST::control_array eps = COST::control_array::Random();
  COST::control_array std_dev = COST::control_array::Constant(0.5);
  float lambda = 0.8;

  /**
   * GPU Setup
   */
  cost.GPUSetup();
  cudaStream_t stream_t;
  cudaStreamCreate(&stream_t);
  float* u_d;
  float* eps_d;
  float* std_dev_d;
  float* GPU_result_d;
  // Allocate Memory on GPU
  size_t control_size = sizeof(float) * COST::CONTROL_DIM;
  HANDLE_ERROR(cudaMalloc((void**)&u_d, control_size));
  HANDLE_ERROR(cudaMalloc((void**)&eps_d, control_size));
  HANDLE_ERROR(cudaMalloc((void**)&std_dev_d, control_size));
  HANDLE_ERROR(cudaMalloc((void**)&GPU_result_d, sizeof(float) * num_results));

  // Copy data to GPU
  HANDLE_ERROR(cudaMemcpyAsync(u_d, u.data(), control_size, cudaMemcpyHostToDevice, stream_t));
  HANDLE_ERROR(cudaMemcpyAsync(eps_d, eps.data(), control_size, cudaMemcpyHostToDevice, stream_t));
  HANDLE_ERROR(cudaMemcpyAsync(std_dev_d, std_dev.data(), control_size, cudaMemcpyHostToDevice, stream_t));
  HANDLE_ERROR(cudaStreamSynchronize(stream_t));

  std::array<float, num_results> CPU_result;
  CPU_result[0] = cost.computeLikelihoodRatioCost(u, eps, std_dev, lambda);
  CPU_result[1] = cost.computeFeedbackCost(u, std_dev, lambda);

  // call GPU Kernel
  std::array<float, num_results> GPU_result;
  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(1, 1, 1);
  ControlCostKernel<COST>
      <<<dimGrid, dimBlock, 0, stream_t>>>(cost.cost_d_, u_d, eps_d, std_dev_d, lambda, GPU_result_d);
  CudaCheckError();
  HANDLE_ERROR(cudaStreamSynchronize(stream_t));
  // Copy GPU results back to Host
  HANDLE_ERROR(
      cudaMemcpyAsync(GPU_result.data(), GPU_result_d, sizeof(float) * num_results, cudaMemcpyDeviceToHost, stream_t));
  HANDLE_ERROR(cudaStreamSynchronize(stream_t));

  array_assert_float_eq<num_results>(CPU_result, GPU_result);
}
