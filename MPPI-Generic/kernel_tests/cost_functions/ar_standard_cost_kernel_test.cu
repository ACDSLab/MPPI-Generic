//#include <cost_functions/autorally/ar_standard_cost.cuh>

template<typename COST_T, typename PARAMS_T>
__global__ void parameterTestKernel(const COST_T* cost, PARAMS_T& params, int& width, int& height) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if (tid == 0) {
    //printf("")
    params = cost->getParams();
    width = cost->getWidth();
    height = cost->getHeight();
  }
}

template<typename COST_T, typename PARAMS_T>
void launchParameterTestKernel(const COST_T& cost, PARAMS_T& params, int& width, int& height) {
  // Allocate memory on the CPU for checking the mass
  PARAMS_T* params_d;
  int* width_d;
  int* height_d;
  HANDLE_ERROR(cudaMalloc((void**)&params_d, sizeof(PARAMS_T)))
  HANDLE_ERROR(cudaMalloc((void**)&width_d, sizeof(float)))
  HANDLE_ERROR(cudaMalloc((void**)&height_d, sizeof(float)))

  parameterTestKernel<COST_T, PARAMS_T><<<1,1>>>(static_cast<COST_T*>(cost.cost_d_), *params_d, *width_d, *height_d);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(&params, params_d, sizeof(PARAMS_T), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&width, width_d, sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&height, height_d, sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(params_d);
}

// TODO actually check texture
__global__ void checkCudaArrayKernel(float4* result_arr, cudaArray* array, int number) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if(tid < number) {
    //printf("The thread id is: %i\n", tid);
    result_arr[tid].x = 0.0;
    result_arr[tid].y = 0.0;
    result_arr[tid].z = 0.0;
    result_arr[tid].w = 0.0;
    //result_arr[tid] = array[tid];
    //printf(array[tid]);
  }
}

void launchCheckCudaArray(std::vector<float4>& result_arr, cudaArray* array, int number) {
  float4* results_d;
  HANDLE_ERROR(cudaMalloc((void**)&results_d, sizeof(float4)*number));

  result_arr.resize(number);

  dim3 threadsPerBlock(4, 1);
  dim3 numBlocks(1, 1);
  checkCudaArrayKernel<<<numBlocks,threadsPerBlock>>>(results_d, array, number);
  CudaCheckError();
  cudaDeviceSynchronize();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(result_arr.data(), results_d, sizeof(float4)*number, cudaMemcpyDeviceToHost));

  cudaFree(results_d);
}

template<typename COST_T>
__global__ void transformTestKernel(float3* results, COST_T* cost) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if (tid == 0) {
    //printf("")
    results[0] = cost->getParams().r_c1;
    results[1] = cost->getParams().r_c2;
    results[2] = cost->getParams().trs;
  }
}

template<typename COST_T>
void launchTransformTestKernel(std::vector<float3>& result, const COST_T& cost) {
  result.resize(3);

  // Allocate memory on the CPU for checking the mass
  float3* results_d;
  HANDLE_ERROR(cudaMalloc((void**)&results_d, sizeof(float3) * 3))

  transformTestKernel<<<1,1>>>(results_d, cost.cost_d_);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(result.data(), results_d, sizeof(float3)*3, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(results_d);
}



template<typename COST_T>
__global__ void textureTestKernel(const COST_T& cost, float4* test_results, float2* test_indexes, int num_points) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if(tid < num_points) {
    //printf("thread id: %i went to check texture at index %i, %i\n", tid, test_indexes[tid].x, test_indexes[tid].y);

    // query texture
    float4 track_params_back = cost.queryTexture(test_indexes[tid].x, test_indexes[tid].y);
    // put result in array
    //printf("thread id: %i got texture point (%f, %f, %f, %f)\n", tid, track_params_back.x, track_params_back.y, track_params_back.z, track_params_back.w);
    test_results[tid] = track_params_back;
    //test_results[tid].x = 1;
  }
}

template<typename COST_T>
void launchTextureTestKernel(const COST_T& cost, std::vector<float4>& test_results, std::vector<float2>& test_indexes) {
  int num_test_points = test_indexes.size();
  test_results.resize(num_test_points);

  float4* tex_results_d;
  float2* tex_test_indexes_d;
  HANDLE_ERROR(cudaMalloc((void**)&tex_results_d, sizeof(float4)*num_test_points))
  HANDLE_ERROR(cudaMalloc((void**)&tex_test_indexes_d, sizeof(float2)*num_test_points))

  HANDLE_ERROR(cudaMemcpy(tex_test_indexes_d, test_indexes.data(), sizeof(float2)*num_test_points, cudaMemcpyHostToDevice));

  // TODO amount should depend on the number of query points
  dim3 threadsPerBlock(num_test_points, 1);
  dim3 numBlocks(1, 1);
  textureTestKernel<<<numBlocks,threadsPerBlock>>>(*cost.cost_d_, tex_results_d, tex_test_indexes_d, num_test_points);
  CudaCheckError();
  cudaDeviceSynchronize();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(test_results.data(), tex_results_d, sizeof(float4)*num_test_points, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  cudaFree(tex_results_d);
  cudaFree(tex_test_indexes_d);
}

template<typename COST_T>
__global__ void textureTransformTestKernel(COST_T& cost, float4* test_results, float2* test_indexes, int num_points) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if(tid < num_points) {
    // query texture
    float4 track_params_back = cost.queryTextureTransformed(test_indexes[tid].x, test_indexes[tid].y);
    // put result in array
    test_results[tid] = track_params_back;
    //test_results[tid].x = 1;
  }
}

template<typename COST_T>
void launchTextureTransformTestKernel(const COST_T& cost, std::vector<float4>& test_results, std::vector<float2>& test_indexes) {
  int num_test_points = test_indexes.size();
  test_results.resize(num_test_points);

  float4* tex_results_d;
  float2* tex_test_indexes_d;
  HANDLE_ERROR(cudaMalloc((void**)&tex_results_d, sizeof(float4)*num_test_points))
  HANDLE_ERROR(cudaMalloc((void**)&tex_test_indexes_d, sizeof(float2)*num_test_points))

  HANDLE_ERROR(cudaMemcpy(tex_test_indexes_d, test_indexes.data(), sizeof(float2)*num_test_points, cudaMemcpyHostToDevice));

  // TODO amount should depend on the number of query points
  dim3 threadsPerBlock(num_test_points, 1);
  dim3 numBlocks(1, 1);
  textureTransformTestKernel<<<numBlocks,threadsPerBlock>>>(*cost.cost_d_, tex_results_d, tex_test_indexes_d, num_test_points);
  CudaCheckError();
  cudaDeviceSynchronize();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(test_results.data(), tex_results_d, sizeof(float4)*num_test_points, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  cudaFree(tex_results_d);
  cudaFree(tex_test_indexes_d);
}

template<typename COST_T>
__global__ void trackCostTestKernel(COST_T* cost, float3* test_indexes, int num_points,
                                    float* cost_results, int* crash_results) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < num_points) {
    float state[7];
    int crash = 0;
    state[0] = test_indexes[tid].x;
    state[1] = test_indexes[tid].y;
    state[2] = test_indexes[tid].z;
    printf("got test indexes %d, state %f, %f, %f\n", tid, state[0], state[1], state[2]);
    cost_results[tid] = cost->getTrackCost(state, &crash);
    printf("set results %d\n", tid);
    crash_results[tid] = crash;
    printf("set crash results %d\n", tid);
  }
}

template<typename COST_T>
void launchTrackCostTestKernel(const COST_T& cost, std::vector<float3>& test_indexes,
                               std::vector<float>& cost_results, std::vector<int>& crash_results) {

  int num_test_points = test_indexes.size();
  crash_results.resize(num_test_points);
  cost_results.resize(num_test_points);

  float* cost_results_d;
  int* crash_results_d;
  float3* test_indexes_d;
  HANDLE_ERROR(cudaMalloc((void**)&cost_results_d, sizeof(float)*num_test_points))
  HANDLE_ERROR(cudaMalloc((void**)&crash_results_d, sizeof(int)*num_test_points))
  HANDLE_ERROR(cudaMalloc((void**)&test_indexes_d, sizeof(float3)*num_test_points))

  HANDLE_ERROR(cudaMemcpy(test_indexes_d, test_indexes.data(), sizeof(float3)*num_test_points, cudaMemcpyHostToDevice));

  // TODO amount should depend on the number of query points
  dim3 threadsPerBlock(num_test_points, 1);
  dim3 numBlocks(1, 1);
  trackCostTestKernel<<<numBlocks,threadsPerBlock>>>(cost.cost_d_, test_indexes_d, num_test_points, cost_results_d, crash_results_d);
  CudaCheckError();
  cudaDeviceSynchronize();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(cost_results.data(), cost_results_d, sizeof(float)*num_test_points, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(crash_results.data(), crash_results_d, sizeof(int)*num_test_points, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  cudaFree(cost_results_d);
  cudaFree(crash_results_d);
  cudaFree(test_indexes_d);
}

template<typename COST_T>
__global__ void computeCostTestKernel(COST_T& cost, float* test_xu, float* cost_results, int num_points) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < num_points) {
    float* state = &test_xu[tid];
    float* control = &test_xu[tid+7];
    int crash = 0;
    float vars[2] = {1,1};
    float du[2] = {0,0};
    cost_results[tid] = cost.computeCost(state, control, du, vars, &crash, tid);
  }
}

template<typename COST_T>
void launchComputeCostTestKernel(const COST_T& cost, std::vector<std::array<float, 9>>& test_xu, std::vector<float>& cost_results) {

  int num_test_points = test_xu.size();
  cost_results.resize(num_test_points*9);

  float* cost_results_d;
  float* test_xu_d;
  HANDLE_ERROR(cudaMalloc((void**)&cost_results_d, sizeof(float)*num_test_points))
  HANDLE_ERROR(cudaMalloc((void**)&test_xu_d, sizeof(float)*9*num_test_points))

  for(int i = 0; i < num_test_points; i++) {
    for(int j = 0; j < 9; j++) {
      cost_results[9*i+j] = test_xu[i][j];
    }
  }

  HANDLE_ERROR(cudaMemcpy(test_xu_d, test_xu.data(), sizeof(float)*9*num_test_points, cudaMemcpyHostToDevice));

  // TODO amount should depend on the number of query points
  dim3 threadsPerBlock(num_test_points, 1);
  dim3 numBlocks(1, 1);
  computeCostTestKernel<<<numBlocks,threadsPerBlock>>>(*cost.cost_d_, test_xu_d, cost_results_d, num_test_points);
  CudaCheckError();
  cudaDeviceSynchronize();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(cost_results.data(), cost_results_d, sizeof(float)*num_test_points, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  cudaFree(cost_results_d);
  cudaFree(test_xu_d);
}

//explicit template instantiation
/*
template __global__ void parameterTestKernel<ARStandardCost<>, ARStandardCostParams>(const ARStandardCost<>* cost, ARStandardCostParams& params, int& width, int& height);

template void launchParameterTestKernel<ARStandardCost<>, ARStandardCostParams>(const ARStandardCost<>& cost, ARStandardCostParams& params, int& width, int& height);

template void launchTransformTestKernel<ARStandardCost<>>(std::vector<float3>& result, const ARStandardCost<>& cost);

template __global__ void transformTestKernel<ARStandardCost<>>(float3* results, ARStandardCost<>* cost);

template void launchTextureTestKernel<ARStandardCost<>>(const ARStandardCost<>& cost, std::vector<float4>& test_results, std::vector<float2>& test_indexes);

template __global__ void textureTestKernel<ARStandardCost<>>(const ARStandardCost<>& cost, float4* test_results, float2* test_indexes, int num_points);

template void launchTextureTransformTestKernel<ARStandardCost<>>(const ARStandardCost<>& cost, std::vector<float4>& test_results, std::vector<float2>& test_indexes);

template __global__ void textureTransformTestKernel<ARStandardCost<>>(ARStandardCost<>& cost, float4* test_results, float2* test_indexes, int num_points);

template void launchTrackCostTestKernel<ARStandardCost<>>(const ARStandardCost<>& cost, std::vector<float3>& test_indexes,
                               std::vector<float>& cost_results, std::vector<int>& crash_results);

template __global__ void trackCostTestKernel<ARStandardCost<>>(ARStandardCost<>* cost, float3* test_indexes, int num_points,
                                    float* cost_results, int* crash_results);

template void launchComputeCostTestKernel<ARStandardCost<>>(const ARStandardCost<>& cost, std::vector<std::array<float, 9>>& test_xu, std::vector<float>& cost_results);

template __global__ void computeCostTestKernel<ARStandardCost<>>(ARStandardCost<>& cost, float* test_xu, float* cost_results, int num_points);
*/
