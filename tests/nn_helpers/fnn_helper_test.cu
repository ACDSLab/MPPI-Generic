#include <gtest/gtest.h>

#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <math.h>

#include <mppi/utils/nn_helpers/fnn_helper.cuh>
// Auto-generated header file
#include <autorally_test_network.h>
#include <mppi/utils/network_helper_kernel_test.cuh>
#include <unsupported/Eigen/NumericalDiff>
#include "mppi/ddp/ddp_dynamics.h"

class FNNHelperTest : public testing::Test
{
protected:
  void SetUp() override
  {
    generator = std::default_random_engine(7.0);
    distribution = std::normal_distribution<float>(0.0, 1.0);
  }

  void TearDown() override
  {
  }

  std::default_random_engine generator;
  std::normal_distribution<float> distribution;
};

TEST_F(FNNHelperTest, ParamsConstructor1)
{
  const int layers = FNNParams<5, 10, 3>::NUM_LAYERS;
  const int padding = FNNParams<5, 10, 3>::PRIME_PADDING;
  const int largest_layer = FNNParams<5, 10, 3>::LARGEST_LAYER;
  const int shared_mem_grd = FNNParams<5, 10, 3>::SHARED_MEM_REQUEST_GRD;
  const int shared_mem_blk = FNNParams<5, 10, 3>::SHARED_MEM_REQUEST_BLK;
  const int input_dim = FNNParams<5, 10, 3>::INPUT_DIM;
  const int output_dim = FNNParams<5, 10, 3>::OUTPUT_DIM;
  EXPECT_EQ(layers, 3);
  EXPECT_EQ(padding, 1);
  EXPECT_EQ(largest_layer, 11);
  EXPECT_EQ(shared_mem_grd, sizeof(FNNParams<5, 10, 3>));
  EXPECT_EQ(shared_mem_blk, 22);
  EXPECT_EQ(input_dim, 5);
  EXPECT_EQ(output_dim, 3);

  FNNParams<5, 10, 3>::ThetaArr theta_arr;
  EXPECT_EQ(sizeof(theta_arr) / sizeof(theta_arr[0]), 93);

  FNNParams<5, 10, 3>::StrideIcsArr stride_arr;
  EXPECT_EQ(sizeof(stride_arr) / sizeof(stride_arr[0]), 4);

  FNNParams<5, 10, 3>::NetStructureArr structure_arr;
  EXPECT_EQ(sizeof(structure_arr) / sizeof(stride_arr[0]), 3);

  FNNParams<5, 10, 3> params;
  auto stride = params.stride_idcs;
  EXPECT_EQ(stride[0], 0);
  EXPECT_EQ(stride[1], 50);
  EXPECT_EQ(stride[2], 60);
  EXPECT_EQ(stride[3], 90);

  auto structure = params.net_structure;
  EXPECT_EQ(structure[0], 5);
  EXPECT_EQ(structure[1], 10);
  EXPECT_EQ(structure[2], 3);

  auto theta = params.theta;
  for (int i = 0; i < 93; i++)
  {
    EXPECT_FLOAT_EQ(theta[i], 0.0f);
  }
}

TEST_F(FNNHelperTest, ParamsConstructor2)
{
  const int layers = FNNParams<5, 10, 20, 3, 3>::NUM_LAYERS;
  const int padding = FNNParams<5, 10, 20, 3, 3>::PRIME_PADDING;
  const int largest_layer = FNNParams<5, 10, 20, 3, 3>::LARGEST_LAYER;
  const int shared_mem_grd = FNNParams<5, 10, 20, 3, 3>::SHARED_MEM_REQUEST_GRD;
  const int shared_mem_blk = FNNParams<5, 10, 20, 3, 3>::SHARED_MEM_REQUEST_BLK;
  const int input_dim = FNNParams<5, 10, 3>::INPUT_DIM;
  const int output_dim = FNNParams<5, 10, 3>::OUTPUT_DIM;
  EXPECT_EQ(layers, 5);
  EXPECT_EQ(padding, 1);
  EXPECT_EQ(largest_layer, 21);
  EXPECT_EQ(shared_mem_grd, sizeof(FNNParams<5, 10, 20, 3, 3>));
  EXPECT_EQ(shared_mem_blk, 42);
  EXPECT_EQ(input_dim, 5);
  EXPECT_EQ(output_dim, 3);

  FNNParams<5, 10, 20, 3, 3>::ThetaArr theta_arr;
  EXPECT_EQ(sizeof(theta_arr) / sizeof(theta_arr[0]), 355);

  FNNParams<5, 10, 20, 3, 3>::StrideIcsArr stride_arr;
  EXPECT_EQ(sizeof(stride_arr) / sizeof(stride_arr[0]), 8);

  FNNParams<5, 10, 20, 3, 3>::NetStructureArr structure_arr;
  EXPECT_EQ(sizeof(structure_arr) / sizeof(stride_arr[0]), 5);

  FNNParams<5, 10, 20, 3, 3> params;
  auto stride = params.stride_idcs;
  EXPECT_EQ(stride[0], 0);
  EXPECT_EQ(stride[1], 50);
  EXPECT_EQ(stride[2], 60);
  EXPECT_EQ(stride[3], 260);
  EXPECT_EQ(stride[4], 280);
  EXPECT_EQ(stride[5], 340);
  EXPECT_EQ(stride[6], 343);
  EXPECT_EQ(stride[7], 352);

  auto structure = params.net_structure;
  EXPECT_EQ(structure[0], 5);
  EXPECT_EQ(structure[1], 10);
  EXPECT_EQ(structure[2], 20);
  EXPECT_EQ(structure[3], 3);
  EXPECT_EQ(structure[4], 3);

  auto theta = params.theta;
  for (int i = 0; i < 355; i++)
  {
    EXPECT_FLOAT_EQ(theta[i], 0.0f);
  }
}

TEST_F(FNNHelperTest, ParamsConstructor3)
{
  const int layers = FNNParams<5, 3>::NUM_LAYERS;
  const int padding = FNNParams<5, 3>::PRIME_PADDING;
  const int largest_layer = FNNParams<5, 3>::LARGEST_LAYER;
  const int shared_mem_grd = FNNParams<5, 3>::SHARED_MEM_REQUEST_GRD;
  const int shared_mem_blk = FNNParams<5, 3>::SHARED_MEM_REQUEST_BLK;
  const int input_dim = FNNParams<5, 10, 3>::INPUT_DIM;
  const int output_dim = FNNParams<5, 10, 3>::OUTPUT_DIM;
  EXPECT_EQ(layers, 2);
  EXPECT_EQ(padding, 1);
  EXPECT_EQ(largest_layer, 6);
  EXPECT_EQ(shared_mem_grd, sizeof(FNNParams<5, 3>));
  EXPECT_EQ(shared_mem_blk, 12);
  EXPECT_EQ(input_dim, 5);
  EXPECT_EQ(output_dim, 3);

  FNNParams<5, 3>::ThetaArr theta_arr;
  EXPECT_EQ(sizeof(theta_arr) / sizeof(theta_arr[0]), 18);

  FNNParams<5, 3>::StrideIcsArr stride_arr;
  EXPECT_EQ(sizeof(stride_arr) / sizeof(stride_arr[0]), 2);

  FNNParams<5, 3>::NetStructureArr structure_arr;
  EXPECT_EQ(sizeof(structure_arr) / sizeof(stride_arr[0]), 2);

  FNNParams<5, 3> params;
  auto stride = params.stride_idcs;
  EXPECT_EQ(stride[0], 0);
  EXPECT_EQ(stride[1], 15);

  auto structure = params.net_structure;
  EXPECT_EQ(structure[0], 5);
  EXPECT_EQ(structure[1], 3);

  auto theta = params.theta;
  for (int i = 0; i < 18; i++)
  {
    EXPECT_FLOAT_EQ(theta[i], 0.0f);
  }
}

TEST_F(FNNHelperTest, BindStream)
{
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  FNNHelper<FNNParams<5, 10, 3>> helper(stream);

  EXPECT_EQ(helper.stream_, stream);
}

TEST_F(FNNHelperTest, GPUSetupAndParamsCheck)
{
  FNNHelper<FNNParams<6, 32, 32, 4>> model;

  std::array<float, 1412> theta = model.getTheta();
  std::array<int, 6> stride = model.getStideIdcs();
  std::array<int, 4> net_structure = model.getNetStructure();

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};
  std::array<float, 1412> shared_theta_result = {};
  std::array<int, 6> shared_stride_result = {};
  std::array<int, 4> shared_net_structure_result = {};

  EXPECT_EQ(model.GPUMemStatus_, false);
  EXPECT_EQ(model.network_d_, nullptr);

  model.GPUSetup();

  EXPECT_EQ(model.GPUMemStatus_, true);
  EXPECT_NE(model.network_d_, nullptr);

  // launch kernel
  launchParameterCheckTestKernel<FNNHelper<FNNParams<6, 32, 32, 4>>, 1412, 6, 4>(
      model, theta_result, stride_result, net_structure_result, shared_theta_result, shared_stride_result,
      shared_net_structure_result);

  for (int i = 0; i < 1412; i++)
  {
    // these are a bunch of mostly random values and nan != nan
    EXPECT_FLOAT_EQ(theta_result[i], theta[i]);
    EXPECT_FLOAT_EQ(shared_theta_result[i], theta[i]);
  }
  for (int i = 0; i < 6; i++)
  {
    EXPECT_EQ(stride_result[i], stride[i]);
    EXPECT_EQ(shared_stride_result[i], stride[i]);
  }

  for (int i = 0; i < 4; i++)
  {
    EXPECT_EQ(net_structure[i], net_structure_result[i]);
    EXPECT_EQ(net_structure[i], shared_net_structure_result[i]);
  }
}

TEST_F(FNNHelperTest, UpdateModelTest)
{
  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  std::array<float, 1412> theta = model.getTheta();
  std::array<int, 6> stride = model.getStideIdcs();
  std::array<int, 4> net_structure = model.getNetStructure();

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};
  std::array<float, 1412> shared_theta_result = {};
  std::array<int, 6> shared_stride_result = {};
  std::array<int, 4> shared_net_structure_result = {};

  model.GPUSetup();

  std::vector<float> theta_vec(1412);
  for (int i = 0; i < 1412; i++)
  {
    theta_vec[i] = distribution(generator);
  }

  model.updateModel({ 6, 32, 32, 4 }, theta_vec);

  // check CPU
  for (int i = 0; i < 1412; i++)
  {
    // these are a bunch of mostly random values and nan != nan
    EXPECT_FLOAT_EQ(model.getTheta()[i], theta_vec[i]);
  }

  // launch kernel
  launchParameterCheckTestKernel<FNNHelper<FNNParams<6, 32, 32, 4>>, 1412, 6, 4>(
      model, theta_result, stride_result, net_structure_result, shared_theta_result, shared_stride_result,
      shared_net_structure_result);

  for (int i = 0; i < 1412; i++)
  {
    // these are a bunch of mostly random values and nan != nan
    EXPECT_FLOAT_EQ(theta_result[i], theta_vec[i]);
    EXPECT_FLOAT_EQ(shared_theta_result[i], theta_vec[i]);
  }
  for (int i = 0; i < 6; i++)
  {
    EXPECT_EQ(stride_result[i], stride[i]);
    EXPECT_EQ(shared_stride_result[i], stride[i]);
  }

  for (int i = 0; i < 4; i++)
  {
    EXPECT_EQ(net_structure[i], net_structure_result[i]);
    EXPECT_EQ(net_structure[i], shared_net_structure_result[i]);
  }
}

TEST_F(FNNHelperTest, LoadModelTest)
{
  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  model.GPUSetup();

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
  std::array<float, 1412> shared_theta_result = {};
  std::array<int, 6> shared_stride_result = {};
  std::array<int, 4> shared_net_structure_result = {};

  // launch kernel
  launchParameterCheckTestKernel<FNNHelper<FNNParams<6, 32, 32, 4>>, 1412, 6, 4>(
      model, theta_result, stride_result, net_structure_result, shared_theta_result, shared_stride_result,
      shared_net_structure_result);

  for (int i = 0; i < 1412; i++)
  {
    EXPECT_FLOAT_EQ(theta_result[i], i) << "failed at index " << i;
    EXPECT_FLOAT_EQ(shared_theta_result[i], i) << "failed at index " << i;
  }
}

TEST_F(FNNHelperTest, LoadModelNPZTest)
{
  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  model.GPUSetup();

  std::string path = mppi::tests::test_load_nn_file;
  if (!fileExists(path))
  {
    std::cerr << "Could not load neural net model at path: " << path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(path);
  model.loadParams(param_dict);

  // check CPU
  for (int i = 0; i < 1412; i++)
  {
    EXPECT_FLOAT_EQ(model.getTheta()[i], i) << "failed at index " << i;
  }

  std::array<float, 1412> theta_result = {};
  std::array<int, 6> stride_result = {};
  std::array<int, 4> net_structure_result = {};
  std::array<float, 1412> shared_theta_result = {};
  std::array<int, 6> shared_stride_result = {};
  std::array<int, 4> shared_net_structure_result = {};

  // launch kernel
  launchParameterCheckTestKernel<FNNHelper<FNNParams<6, 32, 32, 4>>, 1412, 6, 4>(
      model, theta_result, stride_result, net_structure_result, shared_theta_result, shared_stride_result,
      shared_net_structure_result);

  for (int i = 0; i < 1412; i++)
  {
    EXPECT_FLOAT_EQ(shared_theta_result[i], i) << "failed at index " << i;
  }
}

TEST_F(FNNHelperTest, forwardCPU)
{
  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  FNNHelper<FNNParams<6, 32, 32, 4>>::input_array input;
  FNNHelper<FNNParams<6, 32, 32, 4>>::output_array output;

  std::vector<float> theta(1412);
  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({ 6, 32, 32, 4 }, theta);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 33);
  EXPECT_FLOAT_EQ(output[1], 33);
  EXPECT_FLOAT_EQ(output[2], 33);
  EXPECT_FLOAT_EQ(output[3], 33);

  // modify bias
  theta[1408] = 2.0;
  theta[1409] = 3.0;
  theta[1410] = 4.0;
  theta[1411] = 5.0;
  model.updateModel({ 6, 32, 32, 4 }, theta);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 34);
  EXPECT_FLOAT_EQ(output[1], 35);
  EXPECT_FLOAT_EQ(output[2], 36);
  EXPECT_FLOAT_EQ(output[3], 37);

  // modify weight
  theta[1280] = 2.0;
  theta[1312] = 2.0;
  theta[1344] = 2.0;
  theta[1376] = 2.0;
  model.updateModel({ 6, 32, 32, 4 }, theta);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 35);
  EXPECT_FLOAT_EQ(output[1], 36);
  EXPECT_FLOAT_EQ(output[2], 37);
  EXPECT_FLOAT_EQ(output[3], 38);
}

TEST_F(FNNHelperTest, forwardGPU)
{
  const int num_rollouts = 1;

  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  model.GPUSetup();

  std::vector<float> theta(1412);
  std::fill(theta.begin(), theta.end(), 1);
  model.updateModel({ 6, 32, 32, 4 }, theta);

  Eigen::Matrix<float, 6, num_rollouts> inputs;
  inputs = Eigen::Matrix<float, 6, num_rollouts>::Zero();

  std::vector<std::array<float, FNNHelper<FNNParams<6, 32, 32, 4>>::INPUT_DIM>> input_arr(num_rollouts);
  std::vector<std::array<float, FNNHelper<FNNParams<6, 32, 32, 4>>::OUTPUT_DIM>> output_arr(num_rollouts);

  for (int y_dim = 1; y_dim < 16; y_dim++)
  {
    for (int state_index = 0; state_index < num_rollouts; state_index++)
    {
      for (int dim = 0; dim < input_arr[0].size(); dim++)
      {
        input_arr[state_index][dim] = inputs.col(state_index)(dim);
      }
    }

    launchForwardTestKernel<FNNHelper<FNNParams<6, 32, 32, 4>>, 32>(model, input_arr, output_arr, y_dim);
    for (int point = 0; point < num_rollouts; point++)
    {
      FNNHelper<FNNParams<6, 32, 32, 4>>::input_array input = inputs.col(point);
      FNNHelper<FNNParams<6, 32, 32, 4>>::output_array output;

      model.forward(input, output);
      for (int dim = 0; dim < 6; dim++)
      {
        EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
      }
      for (int dim = 0; dim < 4; dim++)
      {
        EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_TRUE(isfinite(output_arr[point][dim]));
        EXPECT_FLOAT_EQ(output(dim), 33);
      }
    }
  }
}

TEST_F(FNNHelperTest, forwardGPUCompare)
{
  const int num_rollouts = 1000;

  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  model.GPUSetup();

  std::vector<float> theta(1412);
  for (int i = 0; i < 1412; i++)
  {
    theta[i] = distribution(generator);
  }
  model.updateModel({ 6, 32, 32, 4 }, theta);

  Eigen::Matrix<float, 6, num_rollouts> inputs;
  inputs = Eigen::Matrix<float, 6, num_rollouts>::Random();

  std::vector<std::array<float, FNNHelper<FNNParams<6, 32, 32, 4>>::INPUT_DIM>> input_arr(num_rollouts);
  std::vector<std::array<float, FNNHelper<FNNParams<6, 32, 32, 4>>::OUTPUT_DIM>> output_arr(num_rollouts);

  for (int y_dim = 1; y_dim < 16; y_dim++)
  {
    for (int state_index = 0; state_index < num_rollouts; state_index++)
    {
      for (int dim = 0; dim < input_arr[0].size(); dim++)
      {
        input_arr[state_index][dim] = inputs.col(state_index)(dim);
      }
    }

    launchForwardTestKernel<FNNHelper<FNNParams<6, 32, 32, 4>>, 32>(model, input_arr, output_arr, y_dim);
    for (int point = 0; point < num_rollouts; point++)
    {
      FNNHelper<FNNParams<6, 32, 32, 4>>::input_array input = inputs.col(point);
      FNNHelper<FNNParams<6, 32, 32, 4>>::output_array output;

      model.forward(input, output);
      for (int dim = 0; dim < 6; dim++)
      {
        EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
      }
      for (int dim = 0; dim < 4; dim++)
      {
        EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_TRUE(isfinite(output_arr[point][dim]));
      }
    }
  }
}

TEST_F(FNNHelperTest, TestComputeGradComputationFinite)
{
  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  std::vector<float> theta(1412);
  for (int i = 0; i < 1412; i++)
  {
    theta[i] = distribution(generator);
  }
  model.updateModel({ 6, 32, 32, 4 }, theta);

  FNNHelper<FNNParams<6, 32, 32, 4>>::dfdx numeric_jac;
  FNNHelper<FNNParams<6, 32, 32, 4>>::dfdx analytic_jac;

  for (int i = 0; i < 1000; i++)
  {
    FNNHelper<FNNParams<6, 32, 32, 4>>::input_array input;
    input = FNNHelper<FNNParams<6, 32, 32, 4>>::input_array::Random();

    model.computeGrad(input, analytic_jac);
    EXPECT_TRUE(analytic_jac.allFinite());
  }
}

TEST_F(FNNHelperTest, TestComputeGradComputationCompare)
{
  GTEST_SKIP();
  FNNHelper<FNNParams<6, 32, 32, 4>> model;
  std::vector<float> theta(1412);
  for (int i = 0; i < 1412; i++)
  {
    theta[i] = distribution(generator);
  }
  model.updateModel({ 6, 32, 32, 4 }, theta);

  FNNHelper<FNNParams<6, 32, 32, 4>>::dfdx numeric_jac;
  FNNHelper<FNNParams<6, 32, 32, 4>>::dfdx analytic_jac;

  FNNHelper<FNNParams<6, 32, 32, 4>>::input_array input;
  input << 1, 2, 3, 4, 5, 6;

  model.computeGrad(input, analytic_jac);

  // numeric_jac = num_diff.df(input, numeric_jac);

  ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n"
                                                       << numeric_jac << "\nAnalytic Jacobian\n"
                                                       << analytic_jac;
}
