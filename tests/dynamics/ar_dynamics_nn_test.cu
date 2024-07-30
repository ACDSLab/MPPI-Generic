//
// Created by jgibson37 on 1/13/20.
//

#include <gtest/gtest.h>
#include <mppi/dynamics/autorally/ar_nn_model.cuh>
#include <mppi/dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>
#include <stdio.h>
#include <math.h>

// Auto-generated header file
#include <autorally_test_network.h>
#include "mppi/ddp/ddp_model_wrapper.h"

/**
 * Note: the analytical solution for the test NN is outlined in the python script
 */

TEST(ARNeuralNetDynamics, verifyTemplateParamters)
{
  int state_dim = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::STATE_DIM;
  EXPECT_EQ(state_dim, 7);

  int control_dim = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::CONTROL_DIM;
  EXPECT_EQ(control_dim, 2);

  int dynamics_dim = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::DYNAMICS_DIM;
  EXPECT_EQ(dynamics_dim, 7 - 3);

  int num_layers = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::NUM_LAYERS;
  EXPECT_EQ(num_layers, 4);

  int prime_padding = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::PRIME_PADDING;
  EXPECT_EQ(prime_padding, 1);

  int largest_layer = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::LARGEST_LAYER;
  EXPECT_EQ(largest_layer, 32 + 1);

  int num_params = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::NUM_PARAMS;
  EXPECT_EQ(num_params, (6 + 1) * 32 + (32 + 1) * 32 + (32 + 1) * 4);

  int shared_mem_request_grd = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::SHARED_MEM_REQUEST_GRD_BYTES;
  EXPECT_EQ(shared_mem_request_grd, sizeof(FNNParams<6, 32, 32, 4>));

  int shared_mem_request_blk = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::SHARED_MEM_REQUEST_BLK_BYTES;
  EXPECT_EQ(shared_mem_request_blk, (32 + 1) * 2);

  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;
  std::array<int, 4> net_structure = model.getNetStructure();

  EXPECT_EQ(net_structure[0], 6);
  EXPECT_EQ(net_structure[1], 32);
  EXPECT_EQ(net_structure[2], 32);
  EXPECT_EQ(net_structure[3], 4);
}

TEST(ARNeuralNetDynamics, BindStreamControlRanges)
{
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  std::array<float2, 2> u_constraint = {};
  u_constraint[0].x = -1.0;
  u_constraint[0].y = 1.0;

  u_constraint[1].x = -2.0;
  u_constraint[1].y = 2.0;
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model(u_constraint, stream);

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(ARNeuralNetDynamics, BindStreamDefaultArgRanges)
{
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model(stream);

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(ARNeuralNetDynamics, ControlRangesSetDefaultCPU)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  std::array<float2, 2> ranges = model.getControlRanges();
  for (int i = 0; i < 2; i++)
  {
    EXPECT_FLOAT_EQ(ranges[0].x, -FLT_MAX);
    EXPECT_FLOAT_EQ(ranges[0].y, FLT_MAX);
  }
}

TEST(ARNeuralNetDynamics, ControlRangesSetCPU)
{
  std::array<float2, 2> u_constraint = {};
  u_constraint[0].x = -1.0;
  u_constraint[0].y = 1.0;

  u_constraint[1].x = -2.0;
  u_constraint[1].y = 2.0;
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model(u_constraint);

  std::array<float2, 2> ranges = model.getControlRanges();
  EXPECT_FLOAT_EQ(ranges[0].x, -1.0);
  EXPECT_FLOAT_EQ(ranges[0].y, 1.0);

  EXPECT_FLOAT_EQ(ranges[1].x, -2.0);
  EXPECT_FLOAT_EQ(ranges[1].y, 2.0);
}

TEST(ARNeuralNetDynamics, stideIdcsSetDefault)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  std::array<int, 6> result = model.getStideIdcs();

  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 192);
  EXPECT_EQ(result[2], 224);
  EXPECT_EQ(result[3], 1248);
  EXPECT_EQ(result[4], 1280);
  EXPECT_EQ(result[5], 1408);
}

TEST(ARNeuralNetDynamics, GPUSetupAndParamsCheck)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  std::array<float, 1412> theta = model.getTheta();
  std::array<int, 6> stride = model.getStideIdcs();
  std::array<int, 4> net_structure = model.getNetStructure();

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};

  EXPECT_EQ(model.GPUMemStatus_, false);
  EXPECT_EQ(model.model_d_, nullptr);
  EXPECT_NE(model.getHelperPtr(), nullptr);

  model.GPUSetup();

  EXPECT_EQ(model.GPUMemStatus_, true);
  EXPECT_NE(model.model_d_, nullptr);

  // launch kernel
  launchParameterCheckTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 1412, 6, 4>(model, theta_result, stride_result,
                                                                                    net_structure_result);

  for (int i = 0; i < 1412; i++)
  {
    // these are a bunch of mostly random values and nan != nan
    if (!isnan(theta[i]))
    {
      EXPECT_FLOAT_EQ(theta_result[i], theta[i]);
    }
  }
  for (int i = 0; i < 6; i++)
  {
    EXPECT_EQ(stride_result[i], stride[i]);
  }

  for (int i = 0; i < 4; i++)
  {
    EXPECT_EQ(net_structure[i], net_structure_result[i]);
  }
}

TEST(ARNeuralNetDynamics, UpdateModelTest)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  std::array<float, 1412> theta = model.getTheta();
  std::array<int, 6> stride = model.getStideIdcs();
  std::array<int, 4> net_structure = model.getNetStructure();

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};

  model.GPUSetup();

  std::vector<float> theta_vec(1412);
  srand(time(NULL));
  for (int i = 0; i < 1412; i++)
  {
    theta_vec[i] = rand();
  }

  model.updateModel({ 6, 32, 32, 4 }, theta_vec);

  // check CPU
  for (int i = 0; i < 1412; i++)
  {
    // these are a bunch of mostly random values and nan != nan
    if (!isnan(theta_vec[i]))
    {
      EXPECT_FLOAT_EQ(model.getTheta()[i], theta_vec[i]);
    }
  }

  // launch kernel
  launchParameterCheckTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 1412, 6, 4>(model, theta_result, stride_result,
                                                                                    net_structure_result);

  for (int i = 0; i < 1412; i++)
  {
    // these are a bunch of mostly random values and nan != nan
    if (!isnan(theta_vec[i]))
    {
      EXPECT_FLOAT_EQ(theta_result[i], theta_vec[i]) << "failed at index " << i;
    }
  }
  for (int i = 0; i < 6; i++)
  {
    EXPECT_EQ(stride_result[i], stride[i]);
  }

  for (int i = 0; i < 4; i++)
  {
    EXPECT_EQ(net_structure[i], net_structure_result[i]);
  }
}

TEST(ARNeuralNetDynamics, LoadModelTest)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;
  model.GPUSetup();

  // TODO procedurally generate a NN in python and save and run like costs
  std::string path = mppi::tests::test_load_nn_file;
  model.loadParams(path);

  // check CPU
  for (int i = 0; i < 1412; i++)
  {
    EXPECT_FLOAT_EQ(model.getTheta()[i], i) << "failed at index " << i;
  }

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};

  // launch kernel
  launchParameterCheckTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 1412, 6, 4>(model, theta_result, stride_result,
                                                                                    net_structure_result);

  for (int i = 0; i < 1412; i++)
  {
    EXPECT_FLOAT_EQ(theta_result[i], i) << "failed at index " << i;
  }
}

TEST(ARNeuralNetDynamics, computeKinematicsTestCPU)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::state_array s(7, 1);
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::state_array s_der(7, 1);

  s(2) = 0.0;  // yaw
  s(4) = 1.0;  // body frame vx
  s(5) = 2.0;  // body frame vy
  s(6) = 0.0;  // yaw dot

  model.computeKinematics(s, s_der);

  EXPECT_FLOAT_EQ(s_der(0), 1.0);
  EXPECT_FLOAT_EQ(s_der(1), 2.0);
  EXPECT_FLOAT_EQ(s_der(2), 0.0);

  s(2) = M_PI / 2;  // yaw
  s(4) = 3.0;       // body frame vx
  s(5) = 5.0;       // body frame vy
  s(6) = 1.0;       // yaw dot

  model.computeKinematics(s, s_der);

  EXPECT_FLOAT_EQ(s_der(0), -5);
  EXPECT_FLOAT_EQ(s_der(1), 3.0);
  EXPECT_FLOAT_EQ(s_der(2), -1.0);
}

TEST(ARNeuralNetDynamics, computeKinematicsTestGPU)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  std::vector<std::array<float, 7>> s(1);
  std::vector<std::array<float, 7>> s_der(1);

  model.GPUSetup();

  for (int y_dim = 1; y_dim < 17; y_dim++)
  {
    s[0] = { 0.0 };
    s[0][2] = 0.0;  // yaw
    s[0][4] = 1.0;  // body frame vx
    s[0][5] = 2.0;  // body frame vy
    s[0][6] = 0.0;  // yaw dot

    s_der[0] = { 0.0 };

    launchComputeKinematicsTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7>(model, s, s_der, y_dim);

    EXPECT_FLOAT_EQ(s_der[0][0], 1.0);
    EXPECT_FLOAT_EQ(s_der[0][1], 2.0);
    EXPECT_FLOAT_EQ(s_der[0][2], 0.0);

    s[0][2] = M_PI / 2;  // yaw
    s[0][4] = 3.0;       // body frame vx
    s[0][5] = 5.0;       // body frame vy
    s[0][6] = 1.0;       // yaw dot

    launchComputeKinematicsTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7>(model, s, s_der, y_dim);

    EXPECT_FLOAT_EQ(s_der[0][0], -5);
    EXPECT_FLOAT_EQ(s_der[0][1], 3.0);
    EXPECT_FLOAT_EQ(s_der[0][2], -1.0);
  }
}

TEST(ARNeuralNetDynamics, updateStateGPUTest)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  std::vector<std::array<float, 7>> s(1);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(1);

  model.GPUSetup();

  for (int j = 1; j < 17; j++)
  {
    s[0] = { 0.0 };
    s[0][2] = 0.0;  // yaw
    s[0][4] = 1.0;  // body frame vx
    s[0][5] = 2.0;  // body frame vy
    s[0][6] = 0.0;  // yaw dot

    s_der[0] = { 0.0 };
    s_der[0][0] = 1.0;
    s_der[0][1] = 2.0;
    s_der[0][2] = 3.0;

    launchUpdateStateTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7>(model, s, s_der, 0.1, j);

    EXPECT_FLOAT_EQ(s_der[0][0], 1);
    EXPECT_FLOAT_EQ(s_der[0][1], 2);
    EXPECT_FLOAT_EQ(s_der[0][2], 3);
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
void compareFiniteDifferenceGradient(CLASS_T& model, Eigen::MatrixXf& s, Eigen::MatrixXf& ds, Eigen::MatrixXf& u,
                                     Eigen::MatrixXf& du)
{
  Eigen::MatrixXf s_2(7, 1);
  s_2 = s + ds;
  Eigen::MatrixXf u_2(2, 1);
  u_2 = u + du;
  Eigen::MatrixXf s_der(7, 1);
  Eigen::MatrixXf s_der_2(7, 1);
  s_der.setZero();
  s_der_2.setZero();

  Eigen::MatrixXf calculated_A(7, 7);
  Eigen::MatrixXf calculated_B(7, 2);

  model.computeDynamics(s_2, u_2, s_der_2);
  model.computeDynamics(s, u, s_der);
  std::cout << "s_der\n" << s_der << std::endl;
  std::cout << "s_der_2\n" << s_der_2 << std::endl;
  std::cout << "s_der_2 - s_der\n" << (s_der_2 - s_der) << std::endl;

  Eigen::MatrixXf A(7, 7);
  Eigen::MatrixXf B(7, 2);

  model.computeGrad(s, u, A, B);
  std::cout << "A = \n" << A << std::endl;
  std::cout << "B = \n" << B << std::endl;

  // compare A
  for (int i = 0; i < 7; i++)
  {
    for (int j = 0; j < 7; j++)
    {
      EXPECT_NEAR(calculated_A(i, j), A(i, j), 0.01) << "failed at index = " << i << ", " << j;
    }
  }

  // compare B
  for (int i = 0; i < 7; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_NEAR(calculated_B(i, j), B(i, j), 0.01) << "failed at index = " << i << ", " << j;
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

TEST(ARNeuralNetDynamics, computeDynamicsCPU)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::state_array s;
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::control_array u;
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::state_array s_der;
  s.setZero();
  s_der.setZero();
  u << 1, -1;

  std::vector<float> theta(1412);

  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({ 6, 32, 32, 4 }, theta);

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

TEST(ARNeuralNetDynamics, computeDynamicsGPU)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;

  std::vector<std::array<float, 7>> s(1);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(1);
  // steering, throttle
  std::vector<std::array<float, 2>> u(1);

  std::vector<float> theta(1412);
  model.GPUSetup();

  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({ 6, 32, 32, 4 }, theta);

  for (int y_dim = 1; y_dim < 17; y_dim++)
  {
    s[0] = { 0 };
    s_der[0] = { 0 };
    u[0] = { 0 };
    u[0][0] = 1.0;
    u[0][1] = -1.0;

    launchComputeDynamicsTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7, 2>(model, s, u, s_der, y_dim);

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
    EXPECT_FLOAT_EQ(s_der[0][3], 33) << "at y_dim " << y_dim;
    EXPECT_FLOAT_EQ(s_der[0][4], 33) << "at y_dim " << y_dim;
    EXPECT_FLOAT_EQ(s_der[0][5], 33) << "at y_dim " << y_dim;
    EXPECT_FLOAT_EQ(s_der[0][6], 33) << "at y_dim " << y_dim;

    EXPECT_FLOAT_EQ(u[0][0], 1.0);
    EXPECT_FLOAT_EQ(u[0][1], -1.0);
  }
}

// TODO compute state deriv CPU
TEST(ARNeuralNetDynamics, computeStateDerivCPU)
{
}

TEST(ARNeuralNetDynamics, computeStateDerivGPU)
{
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model;
  model.GPUSetup();

  std::vector<std::array<float, 7>> s(1);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(1);
  // steering, throttle
  std::vector<std::array<float, 2>> u(1);

  std::vector<float> theta(1412);
  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({ 6, 32, 32, 4 }, theta);

  for (int j = 1; j < 17; j++)
  {
    s[0] = { 0.0 };
    s[0][4] = 1;
    s[0][5] = 2;
    s[0][6] = 3;

    s_der[0] = { 0.0 };
    u[0] = { 0.0 };

    launchComputeStateDerivTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7, 2>(model, s, u, s_der, j);

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

TEST(ARNeuralNetDynamics, full)
{
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

  for (int y_dim = 1; y_dim < 17; y_dim++)
  {
    std::fill(theta.begin(), theta.end(), 0);
    model.updateModel({ 6, 32, 32, 4 }, theta);

    s[0] = { 0.0 };
    s_der[0] = { 0.0 };
    u[0] = { 0.0 };

    launchFullARNNTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7, 2, 1, 1>(model, s, u, s_der, 0.1, y_dim);

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
    model.updateModel({ 6, 32, 32, 4 }, theta);

    launchFullARNNTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7, 2, 1>(model, s, u, s_der, 0.1, y_dim);

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
    EXPECT_FLOAT_EQ(s_der[0][3], 25.371017);
    EXPECT_FLOAT_EQ(s_der[0][4], 25.371017);
    EXPECT_FLOAT_EQ(s_der[0][5], 25.371017);
    EXPECT_FLOAT_EQ(s_der[0][6], 25.371017);

    EXPECT_FLOAT_EQ(u[0][0], 1.0);
    EXPECT_FLOAT_EQ(u[0][1], -2.0);
  }
}

void parseTextIntoDataPointHelper(std::string text, std::array<float, 7>& state, std::array<float, 7>& state_result,
                                  std::array<float, 7>& state_der, std::array<float, 2>& control)
{
  size_t line_pos = 0;
  size_t prev_line_pos = 1;
  int what_var = 0;
  text.append(" *");
  while ((line_pos = text.find("*", prev_line_pos)) != std::string::npos)
  {
    std::string line = text.substr(prev_line_pos, line_pos - prev_line_pos);
    line.append(" ");
    size_t value_pos = 0;
    size_t prev_value_pos = 0;
    int counter = 0;
    while ((value_pos = line.find(" ", prev_value_pos)) != std::string::npos)
    {
      std::string value = line.substr(prev_value_pos, value_pos - prev_value_pos);
      // makes sure it is a number
      if (!value.empty())
      {
        float number = 0;  // = std::stoi(value.substr(0, std::string::npos));
        if (value[0] == '-' && isdigit(value[1]) || isdigit(value[0]))
        {
          number = std::stof(value);
        }
        else
        {
          prev_value_pos = value_pos + 1;
          continue;
        }
        if (what_var == 0)
        {
          state[counter++] = number;
        }
        else if (what_var == 1)
        {
          control[counter++] = number;
        }
        else if (what_var == 2)
        {
          state_der[counter++] = number;
        }
        else if (what_var == 3)
        {
          state_result[counter++] = number;
        }
      }
      prev_value_pos = std::min(value_pos + 1, line.length());
    }
    what_var++;
    prev_line_pos = std::min(line_pos + 1, text.length());
  }
}

/**
 * values grabbed 7/7/20 from melodic-devel in gazebo with
 * model = <param name="model_path" value="$(env AR_MPPI_PARAMS_PATH)/models/autorally_nnet_09_12_2018.npz" />
 *
 * input state 4.264431 -30.974377 -0.955451 -0.028595 3.346700 0.048521 0.315486
 * input control -0.221381 0.089168
 * output state_der 1.971473 -2.704820 -0.315486 0.136986 -0.877249 0.713279 -2.542408
 * output state 4.303861 -31.028473 -0.961760 -0.025856 3.329155 0.062786 0.264637
 *
 * input state 17.818813 -33.751003 2.603801 0.046806 3.690995 -0.162579 0.046056
 * input control 0.247343 0.521264
 * output state_der -3.086702 2.030306 -0.046056 -0.170345 5.059716 -0.795022 1.001305
 * output state 17.757080 -33.710396 2.602880 0.043399 3.792189 -0.178480 0.066082
 *
 * input state 29.102535 -28.311907 0.324347 0.030684 4.354917 -0.250158 -0.181747
 * input control -0.465928 0.304528
 * output state_der 4.207570 1.150753 0.181747 0.285242 0.759887 1.638221 -3.125369
 * output state 29.186687 -28.288891 0.327982 0.036389 4.370114 -0.217393 -0.244254
 *
 * input state -2.603741 -15.197432 2.065675 0.001336 4.745344 0.066507 0.039070
 * input control 0.212821 0.477029
 * output state_der -2.312211 4.144441 -0.039070 -0.049963 4.542547 -1.031745 1.590496
 * output state -2.649985 -15.114543 2.064894 0.000337 4.836195 0.045872 0.070880
 *
 * input state -2.460799 -15.449237 2.066902 0.001312 4.874847 0.114099 0.022956
 * input control -0.126225 -0.288144
 * output state_der -2.420791 4.232839 -0.022956 0.106002 -2.117054 -0.294837 -1.255360
 * output state -2.509215 -15.364580 2.066442 0.003432 4.832506 0.108203 -0.002152
 *
 * input state -9.680823 -7.434339 11.683491 0.428528 -1.655267 -3.769846 -2.194825
 * input control -0.003010 0.147335
 * output state_der -3.963449 -1.114773 2.194825 -0.100577 3.684401 7.052448 2.626748
 * output state -9.760093 -7.456634 11.727387 0.426516 -1.581579 -3.628797 -2.142290
 *
 * input state 20.899069 -37.733856 0.468771 0.109863 6.329141 -0.916158 -0.136938
 * input control 0.281979 0.775381
 * output state_der 6.060292 2.042114 0.136938 -0.325823 4.262759 -1.176095 0.928013
 * output state 21.020275 -37.693012 0.471509 0.103346 6.414396 -0.939680 -0.118377
 */
TEST(ARNeuralNetDynamics, fullComparedToAutoRallyImpl)
{
  std::array<float2, 2> u_constraint = {};
  u_constraint[0].x = -0.99;
  u_constraint[0].y = 0.65;
  u_constraint[1].x = -0.99;
  u_constraint[1].y = 0.99;

  NeuralNetModel<7, 2, 3, 6, 32, 32, 4> model(u_constraint);

  model.GPUSetup();

  int number_queries = 7;

  std::vector<std::array<float, 7>> s(number_queries);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der(number_queries);
  // steering, throttle
  std::vector<std::array<float, 2>> u(number_queries);

  std::vector<std::array<float, 7>> s_result(number_queries);
  // x_dot, y_dot, theta_dot
  std::vector<std::array<float, 7>> s_der_result(number_queries);
  // steering, throttle
  std::vector<std::array<float, 2>> u_result(number_queries);

  parseTextIntoDataPointHelper(
      "* input state 4.264431 -30.974377 -0.955451 -0.028595 3.346700 0.048521 0.315486"
      "* input control -0.221381 0.089168"
      "* output state_der 1.971473 -2.704820 -0.315486 0.136986 -0.877249 0.713279 -2.542408"
      "* output state 4.303861 -31.028473 -0.961760 -0.025856 3.329155 0.062786 0.264637",
      s[0], s_result[0], s_der_result[0], u[0]);
  parseTextIntoDataPointHelper(
      "* input state 17.818813 -33.751003 2.603801 0.046806 3.690995 -0.162579 0.046056"
      "* input control 0.247343 0.521264"
      "* output state_der -3.086702 2.030306 -0.046056 -0.170345 5.059716 -0.795022 1.001305"
      "* output state 17.757080 -33.710396 2.602880 0.043399 3.792189 -0.178480 0.066082",
      s[1], s_result[1], s_der_result[1], u[1]);
  parseTextIntoDataPointHelper(
      "* input state 29.102535 -28.311907 0.324347 0.030684 4.354917 -0.250158 -0.181747"
      "* input control -0.465928 0.304528"
      "* output state_der 4.207570 1.150753 0.181747 0.285242 0.759887 1.638221 -3.125369"
      "* output state 29.186687 -28.288891 0.327982 0.036389 4.370114 -0.217393 -0.244254",
      s[2], s_result[2], s_der_result[2], u[2]);
  parseTextIntoDataPointHelper(
      "* input state -2.603741 -15.197432 2.065675 0.001336 4.745344 0.066507 0.039070"
      "* input control 0.212821 0.477029"
      "* output state_der -2.312211 4.144441 -0.039070 -0.049963 4.542547 -1.031745 1.590496"
      "* output state -2.649985 -15.114543 2.064894 0.000337 4.836195 0.045872 0.070880",
      s[3], s_result[3], s_der_result[3], u[3]);
  parseTextIntoDataPointHelper(
      "* input state -2.460799 -15.449237 2.066902 0.001312 4.874847 0.114099 0.022956"
      "* input control -0.126225 -0.288144"
      "* output state_der -2.420791 4.232839 -0.022956 0.106002 -2.117054 -0.294837 -1.255360"
      "* output state -2.509215 -15.364580 2.066442 0.003432 4.832506 0.108203 -0.002152",
      s[4], s_result[4], s_der_result[4], u[4]);
  parseTextIntoDataPointHelper(
      "* input state -9.680823 -7.434339 11.683491 0.428528 -1.655267 -3.769846 -2.194825"
      "* input control -0.003010 0.147335"
      "* output state_der -3.963449 -1.114773 2.194825 -0.100577 3.684401 7.052448 2.626748"
      "* output state -9.760093 -7.456634 11.727387 0.426516 -1.581579 -3.628797 -2.142290",
      s[5], s_result[5], s_der_result[5], u[5]);
  parseTextIntoDataPointHelper(
      "* input state 20.899069 -37.733856 0.468771 0.109863 6.329141 -0.916158 -0.136938"
      "* input control 0.281979 0.775381"
      "* output state_der 6.060292 2.042114 0.136938 -0.325823 4.262759 -1.176095 0.928013"
      "* output state 21.020275 -37.693012 0.471509 0.103346 6.414396 -0.939680 -0.118377",
      s[6], s_result[6], s_der_result[6], u[6]);

  std::copy(u.begin(), u.end(), u_result.begin());

  std::string path = mppi::tests::old_autorally_network_file;
  model.loadParams(path);

  for (int y_dim = 1; y_dim < 33; y_dim++)
  {
    std::vector<std::array<float, 7>> s_temp(s);
    std::vector<std::array<float, 7>> s_der_temp(s_der);
    std::vector<std::array<float, 2>> u_temp(u);

    launchFullARNNTestKernel<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>, 7, 2, 7>(model, s_temp, u_temp, s_der_temp,
                                                                             1.0 / 50, y_dim);

    for (int point_index = 0; point_index < 1; point_index++)
    {
      for (int dim = 0; dim < 7; dim++)
      {
        EXPECT_NEAR(s_temp[point_index][dim], s_result[point_index][dim], 0.001)
            << "point_index: " << point_index << " dim: " << dim << " y_dim: " << y_dim
            << " diff = " << std::abs(s_temp[point_index][dim] - s_result[point_index][dim]);
        EXPECT_NEAR(s_der_temp[point_index][dim], s_der_result[point_index][dim], 0.001)
            << "point_index: " << point_index << " dim: " << dim << " y_dim: " << y_dim
            << " diff = " << std::abs(s_der_temp[point_index][dim] - s_der_result[point_index][dim]);
      }
      for (int dim = 0; dim < 2; dim++)
      {
        EXPECT_NEAR(u_temp[point_index][dim], u_result[point_index][dim], 0.001)
            << "point_index: " << point_index << " dpoint_indexm: " << dim;
      }
    }
  }
}

class DynamicsDummy : public NeuralNetModel<7, 2, 3, 6, 32, 32, 4>
{
public:
  bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B)
  {
    return false;
  };
};

TEST(ARNeuralNetDynamics, computeGradTest)
{
  std::string path = mppi::tests::old_autorally_network_file;

  Eigen::Matrix<float, NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::STATE_DIM,
                NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::STATE_DIM + NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::CONTROL_DIM>
      numeric_jac;
  Eigen::Matrix<float, NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::STATE_DIM,
                NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::STATE_DIM + NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::CONTROL_DIM>
      analytic_jac;
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::state_array state;
  state << 4.264431, -30.974377, -0.955451, -0.028595, 3.346700, 0.048521, 0.315486;
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::control_array control;
  control << -0.221381, 0.089168;

  auto analytic_grad_model = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>();
  analytic_grad_model.loadParams(path);

  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::dfdx A_analytic = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::dfdx::Zero();
  NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::dfdu B_analytic = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::dfdu::Zero();

  analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);

  auto numerical_grad_model = DynamicsDummy();
  numerical_grad_model.loadParams(path);

  std::shared_ptr<ModelWrapperDDP<DynamicsDummy>> ddp_model =
      std::make_shared<ModelWrapperDDP<DynamicsDummy>>(&numerical_grad_model);

  analytic_jac.leftCols<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::STATE_DIM>() = A_analytic;
  analytic_jac.rightCols<NeuralNetModel<7, 2, 3, 6, 32, 32, 4>::CONTROL_DIM>() = B_analytic;
  numeric_jac = ddp_model->df(state, control);

  ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-1) << "Numeric Jacobian\n"
                                                       << numeric_jac << "\nAnalytic Jacobian\n"
                                                       << analytic_jac;
}
