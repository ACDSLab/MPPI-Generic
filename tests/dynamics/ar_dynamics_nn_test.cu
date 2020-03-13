//
// Created by jgibson37 on 1/13/20.
//

#include <gtest/gtest.h>
#include <dynamics/autorally/ar_nn_model.cuh>
#include <dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>
#include <stdio.h>
#include <math.h>

// Auto-generated header file
#include <autorally_test_network.h>

//Including neural net model
#ifdef MPPI_NNET_USING_CONSTANT_MEM__
__device__ __constant__ float NNET_PARAMS[param_counter(6,32,32,4)];
#endif

/**
 * Note: the analytical solution for the test NN is outlined in the python script
 */

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

  NeuralNetModel<7,2,3,6,32,32,4> model;
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
  NeuralNetModel<7,2,3,6,32,32,4> model(u_constraint, stream);

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}


TEST(ARNeuralNetDynamics, BindStreamDefaultArgRanges) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  NeuralNetModel<7,2,3,6,32,32,4> model(stream);

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(ARNeuralNetDynamics, ControlRangesSetDefaultCPU) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

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
  NeuralNetModel<7,2,3,6,32,32,4> model(u_constraint);

  std::array<float2, 2> ranges = model.getControlRanges();
  EXPECT_FLOAT_EQ(ranges[0].x, -1.0);
  EXPECT_FLOAT_EQ(ranges[0].y, 1.0);

  EXPECT_FLOAT_EQ(ranges[1].x, -2.0);
  EXPECT_FLOAT_EQ(ranges[1].y, 2.0);
}

TEST(ARNeuralNetDynamics, stideIdcsSetDefault) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

  std::array<int, 6> result = model.getStideIdcs();

  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 192);
  EXPECT_EQ(result[2], 224);
  EXPECT_EQ(result[3], 1248);
  EXPECT_EQ(result[4], 1280);
  EXPECT_EQ(result[5], 1408);
}

TEST(ARNeuralNetDynamics, GPUSetupAndParamsCheck) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

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
  NeuralNetModel<7,2,3,6,32,32,4> model;

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
  NeuralNetModel<7,2,3,6,32,32,4> model;
  model.GPUSetup();

  // TODO procedurally generate a NN in python and save and run like costs
  std::string path = mppi::tests::test_load_nn_file;
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

TEST(ARNeuralNetDynamics, computeKinematicsTestCPU) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

  NeuralNetModel<7,2,3,6,32,32,4>::state_array s(7, 1);
  NeuralNetModel<7,2,3,6,32,32,4>::state_array s_der(7, 1);

  s(2) = 0.0; // yaw
  s(4) = 1.0; // body frame vx
  s(5) = 2.0; // body frame vy
  s(6) = 0.0; // yaw dot

  model.computeKinematics(s, s_der);

  EXPECT_FLOAT_EQ(s_der(0), 1.0);
  EXPECT_FLOAT_EQ(s_der(1), 2.0);
  EXPECT_FLOAT_EQ(s_der(2), 0.0);

  s(2) = M_PI/2; // yaw
  s(4) = 3.0; // body frame vx
  s(5) = 5.0; // body frame vy
  s(6) = 1.0; // yaw dot

  model.computeKinematics(s, s_der);

  EXPECT_FLOAT_EQ(s_der(0), -5);
  EXPECT_FLOAT_EQ(s_der(1), 3.0);
  EXPECT_FLOAT_EQ(s_der(2), -1.0);
}

TEST(ARNeuralNetDynamics, computeKinematicsTestGPU) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

  std::vector<std::array<float, 7>> s(1);
  std::vector<std::array<float, 7>> s_der(1);

  model.GPUSetup();

  for(int y_dim = 1; y_dim < 17; y_dim++) {

    s[0] = {0.0};
    s[0][2] = 0.0; // yaw
    s[0][4] = 1.0; // body frame vx
    s[0][5] = 2.0; // body frame vy
    s[0][6] = 0.0; // yaw dot

    s_der[0] = {0.0};

    launchComputeKinematicsTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 7>(model, s, s_der, y_dim);

    EXPECT_FLOAT_EQ(s_der[0][0], 1.0);
    EXPECT_FLOAT_EQ(s_der[0][1], 2.0);
    EXPECT_FLOAT_EQ(s_der[0][2], 0.0);

    s[0][2] = M_PI/2; // yaw
    s[0][4] = 3.0; // body frame vx
    s[0][5] = 5.0; // body frame vy
    s[0][6] = 1.0; // yaw dot

    launchComputeKinematicsTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 7>(model, s, s_der, y_dim);

    EXPECT_FLOAT_EQ(s_der[0][0], -5);
    EXPECT_FLOAT_EQ(s_der[0][1], 3.0);
    EXPECT_FLOAT_EQ(s_der[0][2], -1.0);
  }
}

TEST(ARNeuralNetDynamics, updateStateGPUTest) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

  std::vector<std::array<float, 7>> s(1);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(1);

  model.GPUSetup();

  for(int j = 1; j < 17; j++) {
    s[0] = {0.0};
    s[0][2] = 0.0; // yaw
    s[0][4] = 1.0; // body frame vx
    s[0][5] = 2.0; // body frame vy
    s[0][6] = 0.0; // yaw dot

    s_der[0] = {0.0};
    s_der[0][0] = 1.0;
    s_der[0][1] = 2.0;
    s_der[0][2] = 3.0;

    launchUpdateStateTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 7>(model, s, s_der, 0.1, j);

    EXPECT_FLOAT_EQ(s_der[0][0], 0);
    EXPECT_FLOAT_EQ(s_der[0][1], 0);
    EXPECT_FLOAT_EQ(s_der[0][2], 0);
    EXPECT_FLOAT_EQ(s_der[0][3], 0);
    EXPECT_FLOAT_EQ(s_der[0][4], 0);
    EXPECT_FLOAT_EQ(s_der[0][5], 0);
    EXPECT_FLOAT_EQ(s_der[0][6], 0);

    EXPECT_FLOAT_EQ(s[0][0], 0.1);
    EXPECT_FLOAT_EQ(s[0][1], 0.2);
    EXPECT_FLOAT_EQ(s[0][2], 0.3);
    EXPECT_FLOAT_EQ(s[0][3], 0.0);
    EXPECT_FLOAT_EQ(s[0][4], 1.0);
    EXPECT_FLOAT_EQ(s[0][5], 2.0);
    EXPECT_FLOAT_EQ(s[0][6], 0.0);
  }

}

/**
 *
 * @tparam CLASS_T
 * @param model
 * @param s
 * @param ds
 * @param u
 * @param du
 */
template <class CLASS_T>
void compareFiniteDifferenceGradient(CLASS_T& model, Eigen::MatrixXf& s, Eigen::MatrixXf& ds, Eigen::MatrixXf& u, Eigen::MatrixXf& du) {
  Eigen::MatrixXf s_2(7, 1);
  s_2 = s + ds;
  Eigen::MatrixXf u_2(2,1);
  u_2 = u + du;
  Eigen::MatrixXf s_der(7, 1);
  Eigen::MatrixXf s_der_2(7, 1);
  s_der.setZero();
  s_der_2.setZero();

  Eigen::MatrixXf calculated_A(7,7);
  Eigen::MatrixXf calculated_B(7,2);

  model.computeDynamics(s_2, u_2, s_der_2);
  model.computeDynamics(s, u, s_der);
  std::cout << "s_der\n" << s_der << std::endl;
  std::cout << "s_der_2\n" << s_der_2 << std::endl;
  std::cout << "s_der_2 - s_der\n" << (s_der_2 - s_der) << std::endl;

  Eigen::MatrixXf A(7,7);
  Eigen::MatrixXf B(7,2);

  model.computeGrad(s, u, A, B);
  std::cout << "A = \n" << A << std::endl;
  std::cout << "B = \n" << B << std::endl;

  // compare A
  for(int i = 0; i < 7; i++) {
    for(int j = 0; j < 7; j++) {
      EXPECT_NEAR(calculated_A(i,j), A(i,j), 0.01) << "failed at index = " << i << ", " << j;
    }
  }

  // compare B
  for(int i = 0; i < 7; i++) {
    for(int j = 0; j < 2; j++) {
      EXPECT_NEAR(calculated_B(i,j), B(i,j), 0.01) << "failed at index = " << i << ", " << j;
    }
  }
}

/*
// Note math for analytical solution is in the python script
TEST(ARNeuralNetDynamics, computeGrad) {
  GTEST_SKIP();
  NeuralNetModel<7,2,3,6,32,32,4> model;

  Eigen::MatrixXf s(7, 1);
  Eigen::MatrixXf ds(7, 1);
  Eigen::MatrixXf u(2, 1);
  Eigen::MatrixXf du(2, 1);
  s.setZero();
  ds.setZero();
  u.setZero();
  du.setZero();
  ds << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  std::vector<float> theta(1412);

  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({6, 32, 32, 4}, theta);

  compareFiniteDifferenceGradient(model, s, ds, u, du);

}
 */

TEST(ARNeuralNetDynamics, computeDynamicsCPU) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

  NeuralNetModel<7,2,3,6,32,32,4>::state_array s;
  NeuralNetModel<7,2,3,6,32,32,4>::control_array u;
  NeuralNetModel<7,2,3,6,32,32,4>::state_array s_der;
  s.setZero();
  s_der.setZero();
  u << 1, -1;

  std::vector<float> theta(1412);

  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({6, 32, 32, 4}, theta);

  model.computeDynamics(s, u, s_der);

  EXPECT_FLOAT_EQ(s(0), 0);
  EXPECT_FLOAT_EQ(s(1), 0);
  EXPECT_FLOAT_EQ(s(2), 0);
  EXPECT_FLOAT_EQ(s(3), 0);
  EXPECT_FLOAT_EQ(s(4), 0);
  EXPECT_FLOAT_EQ(s(5), 0);
  EXPECT_FLOAT_EQ(s(6), 0);

  EXPECT_FLOAT_EQ(s_der(0), 0);
  EXPECT_FLOAT_EQ(s_der(1), 0);
  EXPECT_FLOAT_EQ(s_der(2), 0);
  EXPECT_FLOAT_EQ(s_der(3), 33);
  EXPECT_FLOAT_EQ(s_der(4), 33);
  EXPECT_FLOAT_EQ(s_der(5), 33);
  EXPECT_FLOAT_EQ(s_der(6), 33);

  EXPECT_FLOAT_EQ(u(0), 1.0);
  EXPECT_FLOAT_EQ(u(1), -1.0);
}

TEST(ARNeuralNetDynamics, computeDynamicsGPU) {
  NeuralNetModel<7,2,3,6,32,32,4> model;

  std::vector<std::array<float, 7>> s(1);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(1);
  // steering, throttle
  std::vector<std::array<float, 2>> u(1);

  std::vector<float> theta(1412);
  model.GPUSetup();

  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({6, 32, 32, 4}, theta);

  for(int y_dim = 1; y_dim < 17; y_dim++) {
    s[0] = {0};
    s_der[0] = {0};
    u[0] = {0};
    u[0][0] = 1.0;
    u[0][1] = -1.0;

    launchComputeDynamicsTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 7, 2>(model, s, u, s_der, y_dim);

    EXPECT_FLOAT_EQ(s[0][0], 0);
    EXPECT_FLOAT_EQ(s[0][1], 0);
    EXPECT_FLOAT_EQ(s[0][2], 0);
    EXPECT_FLOAT_EQ(s[0][3], 0);
    EXPECT_FLOAT_EQ(s[0][4], 0);
    EXPECT_FLOAT_EQ(s[0][5], 0);
    EXPECT_FLOAT_EQ(s[0][6], 0);

    EXPECT_FLOAT_EQ(s_der[0][0], 0);
    EXPECT_FLOAT_EQ(s_der[0][1], 0);
    EXPECT_FLOAT_EQ(s_der[0][2], 0);
    EXPECT_FLOAT_EQ(s_der[0][3], 33);
    EXPECT_FLOAT_EQ(s_der[0][4], 33);
    EXPECT_FLOAT_EQ(s_der[0][5], 33);
    EXPECT_FLOAT_EQ(s_der[0][6], 33);

    EXPECT_FLOAT_EQ(u[0][0], 1.0);
    EXPECT_FLOAT_EQ(u[0][1], -1.0);
  }

}

// TODO compute state deriv CPU
TEST(ARNeuralNetDynamics, computeStateDerivCPU) {

}

TEST(ARNeuralNetDynamics, computeStateDerivGPU) {
  NeuralNetModel<7,2,3,6,32,32,4> model;
  model.GPUSetup();

  std::vector<std::array<float, 7>> s(1);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(1);
  // steering, throttle
  std::vector<std::array<float, 2>> u(1);

  std::vector<float> theta(1412);
  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({6, 32, 32, 4}, theta);

  for(int j = 1; j < 17; j++) {
    s[0] = {0.0};
    s[0][4] = 1;
    s[0][5] = 2;
    s[0][6] = 3;

    s_der[0] = {0.0};
    u[0] = {0.0};


    launchComputeStateDerivTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 7, 2>(model, s, u, s_der, j);

    EXPECT_FLOAT_EQ(s[0][0], 0);
    EXPECT_FLOAT_EQ(s[0][1], 0);
    EXPECT_FLOAT_EQ(s[0][2], 0);
    EXPECT_FLOAT_EQ(s[0][3], 0);
    EXPECT_FLOAT_EQ(s[0][4], 1);
    EXPECT_FLOAT_EQ(s[0][5], 2);
    EXPECT_FLOAT_EQ(s[0][6], 3);

    EXPECT_FLOAT_EQ(s_der[0][0], 1);
    EXPECT_FLOAT_EQ(s_der[0][1], 2);
    EXPECT_FLOAT_EQ(s_der[0][2], -3);
    EXPECT_FLOAT_EQ(s_der[0][3], 33);
    EXPECT_FLOAT_EQ(s_der[0][4], 33);
    EXPECT_FLOAT_EQ(s_der[0][5], 33);
    EXPECT_FLOAT_EQ(s_der[0][6], 33);

    EXPECT_FLOAT_EQ(u[0][0], 0);
    EXPECT_FLOAT_EQ(u[0][1], 0);
  }
}

TEST(ARNeuralNetDynamics, full) {
  std::array<float2, 2> u_constraint = {};
  u_constraint[0].x = -1.0;
  u_constraint[0].y = 1.0;
  u_constraint[1].x = -2.0;
  u_constraint[1].y = 2.0;

  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model(u_constraint);

  model.GPUSetup();

  std::vector<std::array<float, 7>> s(1);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(1);
  // steering, throttle
  std::vector<std::array<float, 2>> u(1);

  std::vector<float> theta(1412);

  for(int y_dim = 1; y_dim < 17; y_dim++) {
    std::fill(theta.begin(), theta.end(), 0);
    model.updateModel({6, 32, 32, 4}, theta);

    s[0] = {0.0};
    s_der[0] = {0.0};
    u[0] = {0.0};

    launchFullARNNTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7, 2>(model, s, u,
                                                                          s_der, 0.1, y_dim);

    EXPECT_FLOAT_EQ(s[0][0], 0);
    EXPECT_FLOAT_EQ(s[0][1], 0);
    EXPECT_FLOAT_EQ(s[0][2], 0);
    EXPECT_FLOAT_EQ(s[0][3], 0);
    EXPECT_FLOAT_EQ(s[0][4], 0);
    EXPECT_FLOAT_EQ(s[0][5], 0);
    EXPECT_FLOAT_EQ(s[0][6], 0);

    EXPECT_FLOAT_EQ(s_der[0][0], 0);
    EXPECT_FLOAT_EQ(s_der[0][1], 0);
    EXPECT_FLOAT_EQ(s_der[0][2], 0);
    EXPECT_FLOAT_EQ(s_der[0][3], 0);
    EXPECT_FLOAT_EQ(s_der[0][4], 0);
    EXPECT_FLOAT_EQ(s_der[0][5], 0);
    EXPECT_FLOAT_EQ(s_der[0][6], 0);

    EXPECT_FLOAT_EQ(u[0][0], 0);
    EXPECT_FLOAT_EQ(u[0][1], 0);

    u[0][0] = 100;
    u[0][1] = -20;

    std::fill(theta.begin(), theta.end(), 1);
    model.updateModel({6, 32, 32, 4}, theta);

    launchFullARNNTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7, 2>(model, s, u,
                                                                             s_der, 0.1, y_dim);

    EXPECT_FLOAT_EQ(s[0][0], 0);
    EXPECT_FLOAT_EQ(s[0][1], 0);
    EXPECT_FLOAT_EQ(s[0][2], 0);
    EXPECT_FLOAT_EQ(s[0][3], 2.5371017) << "y_dim " << y_dim;
    EXPECT_FLOAT_EQ(s[0][4], 2.5371017);
    EXPECT_FLOAT_EQ(s[0][5], 2.5371017);
    EXPECT_FLOAT_EQ(s[0][6], 2.5371017);

    EXPECT_FLOAT_EQ(s_der[0][0], 0);
    EXPECT_FLOAT_EQ(s_der[0][1], 0);
    EXPECT_FLOAT_EQ(s_der[0][2], 0);
    EXPECT_FLOAT_EQ(s_der[0][3], 0);
    EXPECT_FLOAT_EQ(s_der[0][4], 0);
    EXPECT_FLOAT_EQ(s_der[0][5], 0);
    EXPECT_FLOAT_EQ(s_der[0][6], 0);

    EXPECT_FLOAT_EQ(u[0][0], 1.0);
    EXPECT_FLOAT_EQ(u[0][1], -2.0);
  }
}

