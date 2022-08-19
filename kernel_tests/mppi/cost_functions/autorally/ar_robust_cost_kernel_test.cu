//
// Created by jgibson37 on 2/7/20.
//

template <class CLASS_T>
__global__ void getCostmapCostTestKernel(CLASS_T* cost, float* test_xu, float* cost_results, int num_points)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_points)
  {
    float* state = &test_xu[tid];
    cost_results[tid] = cost->getCostmapCost(state);
  }
}

template <class CLASS_T>
void launchGetCostmapCostTestKernel(CLASS_T& cost, std::vector<std::array<float, 9>>& test_xu,
                                    std::vector<float>& cost_results)
{
  int num_test_points = test_xu.size();
  cost_results.resize(num_test_points * 9);

  float* cost_results_d;
  float* test_xu_d;
  HANDLE_ERROR(cudaMalloc((void**)&cost_results_d, sizeof(float) * num_test_points))
  HANDLE_ERROR(cudaMalloc((void**)&test_xu_d, sizeof(float) * 9 * num_test_points))

  for (int i = 0; i < num_test_points; i++)
  {
    for (int j = 0; j < 9; j++)
    {
      cost_results[9 * i + j] = test_xu[i][j];
    }
  }

  HANDLE_ERROR(cudaMemcpy(test_xu_d, test_xu.data(), sizeof(float) * 9 * num_test_points, cudaMemcpyHostToDevice));

  // TODO amount should depend on the number of query points
  dim3 threadsPerBlock(num_test_points, 1);
  dim3 numBlocks(1, 1);
  getCostmapCostTestKernel<CLASS_T>
      <<<numBlocks, threadsPerBlock>>>(static_cast<CLASS_T*>(cost.cost_d_), test_xu_d, cost_results_d, num_test_points);
  CudaCheckError();
  cudaDeviceSynchronize();

  // Copy the memory back to the host
  HANDLE_ERROR(
      cudaMemcpy(cost_results.data(), cost_results_d, sizeof(float) * num_test_points, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  cudaFree(cost_results_d);
  cudaFree(test_xu_d);
}

template <class CLASS_T>
__global__ void computeCostTestKernel(CLASS_T* cost, float* test_xu, float* cost_results, int num_points)
{
  __shared__ float theta_c[CLASS_T::SHARED_MEM_REQUEST_GRD + CLASS_T::SHARED_MEM_REQUEST_BLK * 1024];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_points)
  {
    float* state = &test_xu[tid];
    float* control = &test_xu[tid + 7];
    float vars[2] = { 1, 1 };
    float du[2] = { 0, 0 };
    float lambda = 1.0;
    float alpha = 0.0;
    int crash_status[1] = { 0 };
    cost->initializeCosts(state, control, theta_c, 0.0, 0.1);
    cost_results[tid] = cost->computeRunningCost(state, control, du, vars, lambda, alpha, tid, theta_c, crash_status);
  }
}

template <class CLASS_T>
void launchComputeCostTestKernel(CLASS_T& cost, std::vector<std::array<float, 9>>& test_xu,
                                 std::vector<float>& cost_results)
{
  int num_test_points = test_xu.size();
  cost_results.resize(num_test_points * 9);

  float* cost_results_d;
  float* test_xu_d;
  HANDLE_ERROR(cudaMalloc((void**)&cost_results_d, sizeof(float) * num_test_points))
  HANDLE_ERROR(cudaMalloc((void**)&test_xu_d, sizeof(float) * 9 * num_test_points))

  for (int i = 0; i < num_test_points; i++)
  {
    for (int j = 0; j < 9; j++)
    {
      cost_results[9 * i + j] = test_xu[i][j];
    }
  }

  HANDLE_ERROR(cudaMemcpy(test_xu_d, test_xu.data(), sizeof(float) * 9 * num_test_points, cudaMemcpyHostToDevice));

  // TODO amount should depend on the number of query points
  dim3 threadsPerBlock(num_test_points, 1);
  dim3 numBlocks(1, 1);
  computeCostTestKernel<CLASS_T>
      <<<numBlocks, threadsPerBlock>>>(static_cast<CLASS_T*>(cost.cost_d_), test_xu_d, cost_results_d, num_test_points);
  CudaCheckError();
  cudaDeviceSynchronize();

  // Copy the memory back to the host
  HANDLE_ERROR(
      cudaMemcpy(cost_results.data(), cost_results_d, sizeof(float) * num_test_points, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  cudaFree(cost_results_d);
  cudaFree(test_xu_d);
}
