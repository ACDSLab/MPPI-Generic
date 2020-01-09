//
// Created by jason on 1/7/20.
//

#include <gtest/gtest.h>
#include <cost_functions/autorally/ar_standard_cost.cuh>
#include <cost_functions/ar_standard_cost_kernel_test.cuh>

TEST(ARStandardCost, Constructor) {
  ARStandardCost cost(4, 5);

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 5);
}

TEST(ARStandardCost, BindStream) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  ARStandardCost cost(1, 2, stream);

  EXPECT_EQ(cost.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(ARStandardCost, SetGetParamsHost) {
  ARStandardCost::ARStandardCostParams params;
  params.desired_speed = 25;
  params.num_timesteps = 100;
  params.r_c1.x = 0;
  params.r_c1.y = 1;
  params.r_c1.z = 2;
  ARStandardCost cost(4, 5);

  cost.setParams(params);
  ARStandardCost::ARStandardCostParams result_params = cost.getParams();

  EXPECT_FLOAT_EQ(params.desired_speed, result_params.desired_speed);
  EXPECT_EQ(params.num_timesteps, result_params.num_timesteps);
  EXPECT_FLOAT_EQ(params.r_c1.x, result_params.r_c1.x);
  EXPECT_FLOAT_EQ(params.r_c1.y, result_params.r_c1.y);
  EXPECT_FLOAT_EQ(params.r_c1.z, result_params.r_c1.z);
}

/*
 * __global__ void objectAllocationTestKernel(ARStandardCost* cost) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  printf("Entering the kernel!\n");
  printf("The thread id is: %i\n", tid);
  if (tid == 0) {
    printf("The cart mass is: %f\n", cost->getParams().desired_speed);
  }
}
 */

TEST(ARStandardCost, GPUSetupAndParamsToDeviceTest) {
  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost(4,8);
  params.desired_speed = 25;
  params.num_timesteps = 100;
  params.r_c1.x = 0;
  params.r_c1.y = 1;
  params.r_c1.z = 2;
  cost.setParams(params);

  EXPECT_EQ(cost.GPUMemStatus_, false);

  cost.GPUSetup();

  EXPECT_EQ(cost.GPUMemStatus_, true);

  float desired_speed;
  int num_timesteps, height, width;
  float3 r_c1;
  launchParameterTestKernel(cost, desired_speed, num_timesteps, r_c1, width, height);

  EXPECT_FLOAT_EQ(desired_speed, 25);
  EXPECT_EQ(num_timesteps, 100);
  EXPECT_FLOAT_EQ(r_c1.x, 0);
  EXPECT_FLOAT_EQ(r_c1.y, 1);
  EXPECT_FLOAT_EQ(r_c1.z, 2);
  EXPECT_EQ(width, 4);
  EXPECT_EQ(height, 8);

  params.desired_speed = 5;
  params.num_timesteps = 50;
  params.r_c1.x = 4;
  params.r_c1.y = 5;
  params.r_c1.z = 6;
  cost.setParams(params);
  cost.paramsToDevice();

  launchParameterTestKernel(cost, desired_speed, num_timesteps, r_c1, height, width);

  EXPECT_FLOAT_EQ(desired_speed, 5);
  EXPECT_EQ(num_timesteps, 50);
  EXPECT_FLOAT_EQ(r_c1.x, 4);
  EXPECT_FLOAT_EQ(r_c1.y, 5);
  EXPECT_FLOAT_EQ(r_c1.z, 6);
}

