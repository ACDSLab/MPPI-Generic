//
// Created by bvlahov3 on 6/15/21.
//

#include <gtest/gtest.h>
#include <mppi/dynamics/LSTM/LSTM_model.cuh>
#include <mppi/dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>
#include <stdio.h>
#include <math.h>

// Auto-generated header file
#include <autorally_test_network.h>

TEST(LSTMDynamicsTest, BindStreamControlRanges) {
  cudaStream_t stream;
  const int CONTROL_DIM = 2;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  std::array<float2, CONTROL_DIM> u_constraint = {};
  u_constraint[0].x = -1.0;
  u_constraint[0].y = 1.0;

  u_constraint[1].x = -2.0;
  u_constraint[1].y = 2.0;
  LSTMModel<7,CONTROL_DIM,3,32> model(u_constraint, stream);

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}