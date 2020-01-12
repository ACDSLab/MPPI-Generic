#include <cost_functions/autorally/ar_standard_cost.cuh>

__global__ void ParameterTestKernel(ARStandardCost* cost, float& desired_speed, int& num_timesteps,
                                    float3& r_c1, int& width, int& height) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if (tid == 0) {
    //printf("")
    desired_speed = cost->getParams().desired_speed;
    num_timesteps = cost->getParams().num_timesteps;
    r_c1 = cost->getParams().r_c1;
    width = cost->getWidth();
    height = cost->getHeight();
  }
}

void launchParameterTestKernel(const ARStandardCost& cost, float& desired_speed, int& num_timesteps,
                               float3& r_c1, int& width, int& height) {
  // Allocate memory on the CPU for checking the mass
  float* desired_speed_d;
  int* num_timesteps_d;
  int* height_d;
  int* width_d;
  float3* r_c1_d;
  HANDLE_ERROR(cudaMalloc((void**)&desired_speed_d, sizeof(float)))
  HANDLE_ERROR(cudaMalloc((void**)&num_timesteps_d, sizeof(int)))
  HANDLE_ERROR(cudaMalloc((void**)&height_d, sizeof(int)))
  HANDLE_ERROR(cudaMalloc((void**)&width_d, sizeof(int)))
  HANDLE_ERROR(cudaMalloc((void**)&r_c1_d, sizeof(float3)))

  ParameterTestKernel<<<1,1>>>(cost.cost_d_, *desired_speed_d, *num_timesteps_d, *r_c1_d, *width_d, *height_d);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(&desired_speed, desired_speed_d, sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&num_timesteps, num_timesteps_d, sizeof(int), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&height, height_d, sizeof(int), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&width, width_d, sizeof(int), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&r_c1, r_c1_d, sizeof(float3), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(desired_speed_d);
  cudaFree(num_timesteps_d);
  cudaFree(r_c1_d);
  cudaFree(height_d);
  cudaFree(width_d);
}

// TODO actually check texture
__global__ void checkCudaArrayKernel(float4* result_arr, cudaArray* array, int number) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if(tid < number) {
    printf("The thread id is: %i\n", tid);
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


__global__ void textureTestKernel(const ARStandardCost& cost, float4* test_results, float2* test_indexes, int num_points) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("\nEntering the kernel!\n");
  //printf("The thread id is: %i\n", tid);
  if(tid < num_points) {
    printf("thread id: %i went to check texture at index %i, %i\n", tid, test_indexes[tid].x, test_indexes[tid].y);

    // query texture
    float4 track_params_back = cost.queryTexture(test_indexes[tid].x, test_indexes[tid].y);
    // put result in array
    printf("thread id: %i got texture point (%f, %f, %f, %f)\n", tid, track_params_back.x, track_params_back.y, track_params_back.z, track_params_back.w);
    test_results[tid] = track_params_back;
    //test_results[tid].x = 1;
  } else {
    printf("thread id: %i did not check texture\n", tid);
  }
}

void launchTextureTestKernel(const ARStandardCost& cost, std::vector<float4>& test_results, std::vector<float2>& test_indexes) {
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

