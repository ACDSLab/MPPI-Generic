//
// Created by jgibson37 on 1/13/20.
//

#include <gtest/gtest.h>
#include <dynamics/autorally/ar_nn_model.cuh>
#include <dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>

TEST(ARNeuralNetDynamics, ControlRangesSetDefaultCPU) {
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt);

 std::array<float2, 2> ranges = model.getControlRanges();
 for(int i = 0; i < 2; i++) {
   EXPECT_FLOAT_EQ(ranges[0].x, -FLT_MAX);
   EXPECT_FLOAT_EQ(ranges[0].y, FLT_MAX);
 }
}

TEST(ARNeuralNetDynamics, ControlRangesSetCPU) {
  std::array<float2, 2> u_constraint = {};
  u_constraint[0].x = -1.0;
  u_constraint[0].y = 1.0;

  u_constraint[1].x = -2.0;
  u_constraint[1].y = 2.0;
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt, u_constraint);

  std::array<float2, 2> ranges = model.getControlRanges();
  EXPECT_FLOAT_EQ(ranges[0].x, -1.0);
  EXPECT_FLOAT_EQ(ranges[0].y, 1.0);


  EXPECT_FLOAT_EQ(ranges[1].x, -2.0);
  EXPECT_FLOAT_EQ(ranges[1].y, 2.0);
}


