//
// Created by jason on 1/7/20.
//

#include <gtest/gtest.h>
#include <cost_functions/autorally/ar_standard_cost.cuh>
#include <cost_functions/ar_standard_cost_kernel_test.cuh>

TEST(ARStandardCost, Constructor) {
  ARStandardCost cost;
}

TEST(ARStandardCost, BindStream) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  ARStandardCost cost(stream);

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
  ARStandardCost cost;

  cost.setParams(params);
  ARStandardCost::ARStandardCostParams result_params = cost.getParams();

  EXPECT_FLOAT_EQ(params.desired_speed, result_params.desired_speed);
  EXPECT_EQ(params.num_timesteps, result_params.num_timesteps);
  EXPECT_FLOAT_EQ(params.r_c1.x, result_params.r_c1.x);
  EXPECT_FLOAT_EQ(params.r_c1.y, result_params.r_c1.y);
  EXPECT_FLOAT_EQ(params.r_c1.z, result_params.r_c1.z);
}

TEST(ARStandardCost, GPUSetupAndParamsToDeviceTest) {
  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost;
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
  // neither should be set by this sequence
  EXPECT_EQ(width, -1);
  EXPECT_EQ(height, -1);

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

  // neither should be set by this sequence
  EXPECT_EQ(width, -1);
  EXPECT_EQ(height, -1);
}

TEST(ARStandardCost, clearCostmapCPUTestValidInputs) {
  ARStandardCost cost;
  cost.clearCostmapCPU(4, 8);

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);

  for(int i = 0; i < 4 * 8; i++) {
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).x, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).y, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).z, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).w, 0);
  }
}

TEST(ARStandardCost, clearCostmapCPUTestDefault) {
  ARStandardCost cost;
  cost.clearCostmapCPU(4, 8);
  cost.clearCostmapCPU();

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);

  for(int i = 0; i < 4 * 8; i++) {
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).x, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).y, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).z, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).w, 0);
  }
}


TEST(ARStandardCost, clearCostmapCPUTestDefaultFail) {
  ARStandardCost cost;
  cost.clearCostmapCPU();

  EXPECT_EQ(cost.getWidth(), -1);
  EXPECT_EQ(cost.getHeight(), -1);
}

TEST(ARStandardCost, LoadTrackDataTest) {
  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost;
  // TODO set parameters for cost map
  cost.setParams(params);
  cost.GPUSetup();
  float desired_speed;
  int num_timesteps, height, width;
  float3 r_c1;
  launchParameterTestKernel(cost, desired_speed, num_timesteps, r_c1, width, height);

  // load


}

