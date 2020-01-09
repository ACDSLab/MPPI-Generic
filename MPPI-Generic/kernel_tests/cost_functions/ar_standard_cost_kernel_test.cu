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

  ParameterTestKernel<<<1,1>>>(cost.cost_device_, *desired_speed_d, *num_timesteps_d, *r_c1_d, *width_d, *height_d);
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
