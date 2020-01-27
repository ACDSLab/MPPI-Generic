//
// Created by jgibson37 on 1/13/20.
//

#include <gtest/gtest.h>
#include <dynamics/autorally/ar_nn_model.cuh>
#include <dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>
#include <stdio.h>
#include <math.h>

TEST(ARNeuralNetDynamics, verifyTemplateParamters) {
  int state_dim = NeuralNetModel<7,2,3,6,32,32,4>::STATE_DIM;
  EXPECT_EQ(state_dim, 7);

  int control_dim = NeuralNetModel<7,2,3,6,32,32,4>::CONTROL_DIM;
  EXPECT_EQ(control_dim, 2);

  int dynamics_dim = NeuralNetModel<7,2,3,6,32,32,4>::DYNAMICS_DIM;
  EXPECT_EQ(dynamics_dim, 7-3);

  int num_layers = NeuralNetModel<7,2,3,6,32,32,4>::NUM_LAYERS;
  EXPECT_EQ(num_layers, 4);

  int prime_padding = NeuralNetModel<7,2,3,6,32,32,4>::PRIME_PADDING;
  EXPECT_EQ(prime_padding, 1);

  int largest_layer = NeuralNetModel<7,2,3,6,32,32,4>::LARGEST_LAYER;
  EXPECT_EQ(largest_layer, 32+1);

  int num_params = NeuralNetModel<7,2,3,6,32,32,4>::NUM_PARAMS;
  EXPECT_EQ(num_params, (6+1)*32+(32+1)*32+(32+1)*4);

  int shared_mem_request_grd = NeuralNetModel<7,2,3,6,32,32,4>::SHARED_MEM_REQUEST_GRD;
  EXPECT_EQ(shared_mem_request_grd, 0);

  int shared_mem_request_blk = NeuralNetModel<7,2,3,6,32,32,4>::SHARED_MEM_REQUEST_BLK;
  EXPECT_EQ(shared_mem_request_blk, (32+1)*2);

  NeuralNetModel<7,2,3,6,32,32,4> model(0.1);
  std::array<int, 4> net_structure = model.getNetStructure();

  EXPECT_EQ(net_structure[0], 6);
  EXPECT_EQ(net_structure[1], 32);
  EXPECT_EQ(net_structure[2], 32);
  EXPECT_EQ(net_structure[3], 4);
}

TEST(ARNeuralNetDynamics, BindStreamControlRanges) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  std::array<float2, 2> u_constraint = {};
  u_constraint[0].x = -1.0;
  u_constraint[0].y = 1.0;

  u_constraint[1].x = -2.0;
  u_constraint[1].y = 2.0;
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt, u_constraint, stream);

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}


TEST(ARNeuralNetDynamics, BindStreamDefaultArgRanges) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt, stream);

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(ARNeuralNetDynamics, ControlRangesSetDefaultCPU) {
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt);

 std::array<float2, 2> ranges = model.getControlRanges();
 EXPECT_FLOAT_EQ(model.dt_, 0.1);
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

  EXPECT_FLOAT_EQ(model.dt_, 0.1);

  std::array<float2, 2> ranges = model.getControlRanges();
  EXPECT_FLOAT_EQ(ranges[0].x, -1.0);
  EXPECT_FLOAT_EQ(ranges[0].y, 1.0);

  EXPECT_FLOAT_EQ(ranges[1].x, -2.0);
  EXPECT_FLOAT_EQ(ranges[1].y, 2.0);
}

TEST(ARNeuralNetDynamics, stideIdcsSetDefault) {
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt);

  std::array<int, 6> result = model.getStideIdcs();

  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 192);
  EXPECT_EQ(result[2], 224);
  EXPECT_EQ(result[3], 1248);
  EXPECT_EQ(result[4], 1280);
  EXPECT_EQ(result[5], 1408);
}

TEST(ARNeuralNetDynamics, GPUSetupAndParamsCheck) {
  NeuralNetModel<7,2,3,6,32,32,4> model(0.1);

  std::array<float, 1412> theta = model.getTheta();
  std::array<int, 6> stride = model.getStideIdcs();
  std::array<int, 4> net_structure = model.getNetStructure();

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};

  EXPECT_EQ(model.GPUMemStatus_, false);
  EXPECT_EQ(model.model_d_, nullptr);

  model.GPUSetup();

  EXPECT_EQ(model.GPUMemStatus_, true);
  EXPECT_NE(model.model_d_, nullptr);

  //launch kernel
  launchParameterCheckTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 1412, 6, 4>
          (model, theta_result, stride_result, net_structure_result);

  for(int i = 0; i < 1412; i++) {
    // these are a bunch of mostly random values and nan != nan
    if(!isnan(theta[i])) {
      EXPECT_FLOAT_EQ(theta_result[i], theta[i]);
    }
  }
  for(int i = 0; i < 6; i++) {
    EXPECT_EQ(stride_result[i], stride[i]);
  }

  for(int i = 0; i < 4; i++) {
    EXPECT_EQ(net_structure[i], net_structure_result[i]);
  }
}

TEST(ARNeuralNetDynamics, UpdateModelTest) {
  NeuralNetModel<7,2,3,6,32,32,4> model(0.1);

  std::array<float, 1412> theta = model.getTheta();
  std::array<int, 6> stride = model.getStideIdcs();
  std::array<int, 4> net_structure = model.getNetStructure();

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};

  model.GPUSetup();

  std::vector<float> theta_vec(1412);
  srand (time(NULL));
  for(int i = 0; i < 1412; i++) {
    theta_vec[i] = rand();
  }

  model.updateModel({6, 32, 32, 4}, theta_vec);

  // check CPU
  for(int i = 0; i < 1412; i++) {
    // these are a bunch of mostly random values and nan != nan
    if(!isnan(theta_vec[i])) {
      EXPECT_FLOAT_EQ(model.getTheta()[i], theta_vec[i]);
    }
  }

  //launch kernel
  launchParameterCheckTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 1412, 6, 4>
          (model, theta_result, stride_result, net_structure_result);

  for(int i = 0; i < 1412; i++) {
    // these are a bunch of mostly random values and nan != nan
    if(!isnan(theta_vec[i])) {
      EXPECT_FLOAT_EQ(theta_result[i], theta_vec[i]) << "failed at index " << i;
    }
  }
  for(int i = 0; i < 6; i++) {
    EXPECT_EQ(stride_result[i], stride[i]);
  }

  for(int i = 0; i < 4; i++) {
    EXPECT_EQ(net_structure[i], net_structure_result[i]);
  }
}

TEST(ARNeuralNetDynamics, LoadModelTest) {
  NeuralNetModel<7,2,3,6,32,32,4> model(0.1);
  model.GPUSetup();

  // TODO procedurally generate a NN in python and save and run like costs
  std::string path = "/home/mgandhi3/git/MPPI-Generic/test_nn.npz";
  model.loadParams(path);

  // check CPU
  for(int i = 0; i < 1412; i++) {
    EXPECT_FLOAT_EQ(model.getTheta()[i], i) << "failed at index " << i;
  }

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};

  //launch kernel
  launchParameterCheckTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 1412, 6, 4>
          (model, theta_result, stride_result, net_structure_result);

  for(int i = 0; i < 1412; i++) {
    EXPECT_FLOAT_EQ(theta_result[i], i) << "failed at index " << i;
  }
}

TEST(ARNeuralNetDynamics, enforceConstraintsTest) {
  std::array<float2, 2> u_constraint = {};
  u_constraint[0].x = -1.0;
  u_constraint[0].y = 1.0;

  u_constraint[1].x = -2.0;
  u_constraint[1].y = 2.0;
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt, u_constraint);

  float s[7];
  float u[2];

  u[0] = 10;
  u[1] = 1000;

  model.enforceConstraints(s, u);

  EXPECT_FLOAT_EQ(u[0], 1);
  EXPECT_FLOAT_EQ(u[1], 2);

  u[0] = -124;
  u[1] = -512789;

  model.enforceConstraints(s, u);

  EXPECT_FLOAT_EQ(u[0], -1);
  EXPECT_FLOAT_EQ(u[1], -2);

  u[0] = 0.5;
  u[1] = 1;

  model.enforceConstraints(s, u);

  EXPECT_FLOAT_EQ(u[0], 0.5);
  EXPECT_FLOAT_EQ(u[1], 1.0);
}

TEST(ARNeuralNetDynamics, computeKinematicsTest) {
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt);

  float s[7];
  // x_dot, y_dot, theta_dot
  float s_der[3];

  s[2] = 0.0; // yaw
  s[4] = 1.0; // body frame vx
  s[5] = 2.0; // body frame vy
  s[6] = 0.0; // yaw dot

  model.computeKinematics(s, s_der);

  EXPECT_FLOAT_EQ(s_der[0], 1.0);
  EXPECT_FLOAT_EQ(s_der[1], 2.0);
  EXPECT_FLOAT_EQ(s_der[2], 0.0);

  s[2] = M_PI/2; // yaw
  s[4] = 3.0; // body frame vx
  s[5] = 5.0; // body frame vy
  s[6] = 1.0; // yaw dot

  model.computeKinematics(s, s_der);

  EXPECT_FLOAT_EQ(s_der[0], -5);
  EXPECT_FLOAT_EQ(s_der[1], 3.0);
  EXPECT_FLOAT_EQ(s_der[2], -1.0);
}

TEST(ARNeuralNetDynamics, incrementState) {
  float dt = 0.1;
  NeuralNetModel<7,2,3,6,32,32,4> model(dt);

  std::array<float, 7> s = {0.0};
  // x_dot, y_dot, theta_dot
  std::array<float, 7> s_der = {0.0};

  s[2] = 0.0; // yaw
  s[4] = 1.0; // body frame vx
  s[5] = 2.0; // body frame vy
  s[6] = 0.0; // yaw dot

  s_der[0] = 1.0;
  s_der[1] = 2.0;
  s_der[2] = 3.0;

  model.GPUSetup();

  launchIncrementStateTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 1, 7>(model, s, s_der);

  EXPECT_FLOAT_EQ(s_der[0], 0);
  EXPECT_FLOAT_EQ(s_der[1], 0);
  EXPECT_FLOAT_EQ(s_der[2], 0);

  EXPECT_FLOAT_EQ(s[0], 0.1);
  EXPECT_FLOAT_EQ(s[1], 0.2);
  EXPECT_FLOAT_EQ(s[2], 0.3);
}

TEST(ARNeuralNetDynamics, computeDynamicsTest) {

}
