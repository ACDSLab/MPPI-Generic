//
// Created by jgibson37 on 2/7/20.
//

#include <gtest/gtest.h>
#include <cost_functions/autorally/ar_robust_cost.cuh>
#include <cost_functions/ar_robust_cost_kernel_test.cuh>

// Auto-generated header file
#include <autorally_test_map.h>

TEST(ARRobustCost, Constructor) {
  ARRobustCost<> cost;
}

TEST(ARRobustCost, GPUSetup) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  ARRobustCost<> cost(stream);

  EXPECT_EQ(cost.stream_, stream) << "Stream binding failure.";

  EXPECT_EQ(cost.GPUMemStatus_, false);
  EXPECT_EQ(cost.cost_d_, nullptr);

  cost.GPUSetup();

  EXPECT_EQ(cost.GPUMemStatus_, true);
  EXPECT_NE(cost.cost_d_, nullptr);

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

void checkParameters(ARRobustCostParams& params, ARRobustCostParams& result) {
  EXPECT_FLOAT_EQ(params.speed_coeff, result.speed_coeff);
  EXPECT_FLOAT_EQ(params.track_coeff,result.track_coeff);
  EXPECT_FLOAT_EQ(params.heading_coeff, result.heading_coeff);
  EXPECT_FLOAT_EQ(params.steering_coeff,result.steering_coeff);
  EXPECT_FLOAT_EQ(params.throttle_coeff,result.throttle_coeff);
  EXPECT_FLOAT_EQ(params.slip_coeff,result.slip_coeff);
  EXPECT_FLOAT_EQ(params.crash_coeff,result.crash_coeff);
  EXPECT_FLOAT_EQ(params.boundary_threshold,result.boundary_threshold);
  EXPECT_FLOAT_EQ(params.max_slip_ang,result.max_slip_ang);
  EXPECT_FLOAT_EQ(params.track_slop,result.track_slop);
  EXPECT_FLOAT_EQ(params.num_timesteps,result.num_timesteps);
  EXPECT_FLOAT_EQ(params.r_c1.x,result.r_c1.x);
  EXPECT_FLOAT_EQ(params.r_c1.y,result.r_c1.y);
  EXPECT_FLOAT_EQ(params.r_c1.z,result.r_c1.z);
  EXPECT_FLOAT_EQ(params.r_c2.x,result.r_c2.x);
  EXPECT_FLOAT_EQ(params.r_c2.y,result.r_c2.y);
  EXPECT_FLOAT_EQ(params.r_c2.z,result.r_c2.z);
  EXPECT_FLOAT_EQ(params.trs.x,result.trs.x);
  EXPECT_FLOAT_EQ(params.trs.y,result.trs.y);
  EXPECT_FLOAT_EQ(params.trs.z,result.trs.z);
}

TEST(ARRobustCost, setParams) {
  ARRobustCostParams params;

  params.speed_coeff = 1.0;
  params.track_coeff = 2.0;
  params.heading_coeff = 3.0;
  params.steering_coeff = 4.0;
  params.throttle_coeff = 5.0;
  params.slip_coeff = 6.0;
  params.crash_coeff = 7.0;
  params.boundary_threshold = 8.0;
  params.max_slip_ang = 9.0;
  params.track_slop = 10.0;
  params.num_timesteps = 11;
  params.r_c1.x = 12;
  params.r_c1.y = 13;
  params.r_c1.z = 14;
  params.r_c2.x = 15;
  params.r_c2.y = 16;
  params.r_c2.z = 17;
  params.trs.x = 18;
  params.trs.y = 19;
  params.trs.z = 20;

  ARRobustCost<> cost;

  cost.setParams(params);
  ARRobustCostParams result = cost.getParams();
  checkParameters(params, result);

  cost.GPUSetup();
  int width, height;
  //launchParameterTestKernel<ARRobustCost<ARRobustCostParams>,ARRobustCostParams>(cost, params, width, height);
}

TEST(ARRobustCost, getSpeedCostMapTest) {

}

TEST(ARRobustCost, getSpeedCostNotMapTest) {

}

TEST(ARRobustCost, getStabilizingCostTest) {

}

TEST(ARRobustCost, getCostmapCostTest) {

}

TEST(ARRobustCost, computeCostTest) {

}
