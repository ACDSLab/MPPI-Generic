#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/racer_dubins/racer_dubins_elevation_lstm_unc.cuh>
#include <kernel_tests/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <racer_test_networks.h>
#include <cuda_runtime.h>
#include <mppi/core/mppi_common_new.cuh>

class RacerDubinsElevationLSTMUncertaintyTest : public ::testing::Test
{
public:
  cudaStream_t stream;
  cudaExtent extent = make_cudaExtent(10, 20, 0);
  float map_resolution = 10.0f;          // [m / px]
  float3 origin = make_float3(1, 2, 3);  // [m, m, m]

  std::vector<float> data_vec;
  std::vector<float4> normal_vec;

  void SetUp() override
  {
    CudaCheckError();
    HANDLE_ERROR(cudaStreamCreate(&stream));

    steer_config.init_config.input_dim = 3;
    steer_config.init_config.hidden_dim = 20;
    steer_config.init_config.output_layers = { 23, 100, 8 };
    steer_config.pred_config.input_dim = 4;
    steer_config.pred_config.hidden_dim = 4;
    steer_config.pred_config.output_layers = { 8, 20, 1 };
    steer_config.init_len = 11;

    unc_config.init_config.input_dim = 10;
    unc_config.init_config.hidden_dim = 20;
    unc_config.init_config.output_layers = { 30, 100, 8 };
    unc_config.pred_config.input_dim = 13;
    unc_config.pred_config.hidden_dim = 4;
    unc_config.pred_config.output_layers = { 17, 20, 5 };
    unc_config.init_len = 11;

    mean_config.init_config.input_dim = 10;
    mean_config.init_config.hidden_dim = 20;
    mean_config.init_config.output_layers = { 30, 100, 8 };
    mean_config.pred_config.input_dim = 12;
    mean_config.pred_config.hidden_dim = 4;
    mean_config.pred_config.output_layers = { 16, 20, 2 };
    mean_config.init_len = 11;

    data_vec.resize(extent.width * extent.height);
    normal_vec.resize(extent.width * extent.height);
    for (int i = 0; i < data_vec.size(); i++)
    {
      data_vec[i] = i * 1.0f;
      Eigen::Vector4f random_normal = Eigen::Vector4f::Random();
      normal_vec[i] = make_float4(random_normal.x(), random_normal.y(), random_normal.z(), random_normal.w());
    }
  }

  void TearDown() override
  {
    CudaCheckError();
    HANDLE_ERROR(cudaStreamDestroy(stream));
  }

  LSTMLSTMConfig steer_config;
  LSTMLSTMConfig mean_config;
  LSTMLSTMConfig unc_config;
};

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, Template)
{
  auto dynamics = RacerDubinsElevationLSTMUncertainty(steer_config, mean_config, unc_config);
  EXPECT_EQ(24, RacerDubinsElevationLSTMUncertainty::STATE_DIM);
  EXPECT_EQ(2, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM);
  EXPECT_TRUE(dynamics.checkRequiresBuffer());
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_NE(dynamics.getHelper(), nullptr);
  EXPECT_NE(dynamics.getUncertaintyHelper(), nullptr);
  EXPECT_NE(dynamics.getMeanHelper(), nullptr);
}

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, BindStreamConstructor)
{
  auto dynamics = RacerDubinsElevationLSTMUncertainty(steer_config, mean_config, unc_config, stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->stream_, stream);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->stream_, stream);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->stream_, stream);
}

// TEST_F(RacerDubinsElevationLSTMUncertaintyTest, Deconstructor)
// {
//   {
//     auto dynamics = RacerDubinsElevationLSTMUncertainty(steer_config, mean_config, unc_config);
//     dynamics = RacerDubinsElevationLSTMUncertainty(steer_config, mean_config, unc_config);
//   }
// }

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, BindStream)
{
  auto dynamics = RacerDubinsElevationLSTMUncertainty(steer_config, mean_config, unc_config);
  dynamics.bindToStream(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->stream_, stream);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->stream_, stream);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->stream_, stream);
}

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, GPUSetup)
{
  auto dynamics = RacerDubinsElevationLSTMUncertainty(steer_config, mean_config, unc_config, stream);

  dynamics.GPUSetup();

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->stream_, stream);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->stream_, stream);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->stream_, stream);

  EXPECT_EQ(dynamics.GPUMemStatus_, true);
  EXPECT_NE(dynamics.model_d_, nullptr);

  // steering model
  EXPECT_NE(dynamics.network_d_, nullptr);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->GPUMemStatus_, true);
  EXPECT_NE(dynamics.getHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.getHelper()->getLSTMModel()->network_d_, nullptr);

  // mean model
  EXPECT_NE(dynamics.mean_d_, nullptr);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->GPUMemStatus_, true);
  EXPECT_NE(dynamics.getMeanHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.getMeanHelper()->getLSTMModel()->network_d_, nullptr);

  // mean model
  EXPECT_NE(dynamics.uncertainty_d_, nullptr);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->GPUMemStatus_, true);
  EXPECT_NE(dynamics.getUncertaintyHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.getUncertaintyHelper()->getLSTMModel()->network_d_, nullptr);
}

// TODO test case that checks that I get the same as base class here

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestLoadParams)
{
  CudaCheckError();
  using DYN = RacerDubinsElevationLSTMUncertainty;
  RacerDubinsElevationLSTMUncertainty dynamics =
      RacerDubinsElevationLSTMUncertainty(mppi::tests::racer_dubins_elevation_uncertainty_test);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 17.590296);
  EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 3.286375);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_constant, 9.30152702);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_drag_constant, -0.60327667);

  EXPECT_FLOAT_EQ(dynamics.getParams().c_t[0], 3.0364573);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_t[1], 4.59772491);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_t[2], 4.06954288);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_b[0], 2.56373692);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_b[1], 6.18813848);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_b[2], 25.52443695);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_v[0], 4.39438224e-01);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_v[1], 2.37689335e-02);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_v[2], 2.12573977e-05);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_angle_scale, -13.25719738);
  EXPECT_FLOAT_EQ(dynamics.getParams().unc_scale[0], 0.5);
  EXPECT_FLOAT_EQ(dynamics.getParams().unc_scale[1], 0.2);
  EXPECT_FLOAT_EQ(dynamics.getParams().unc_scale[2], 0.5);
  EXPECT_FLOAT_EQ(dynamics.getParams().unc_scale[3], 0.02);
  EXPECT_FLOAT_EQ(dynamics.getParams().unc_scale[4], 0.02);
  EXPECT_FLOAT_EQ(dynamics.getParams().unc_scale[5], 0.1);
  EXPECT_FLOAT_EQ(dynamics.getParams().unc_scale[6], 0.1);

  EXPECT_EQ(dynamics.getHelper()->getInitModel()->getInputDim(), 3);
  EXPECT_EQ(dynamics.getHelper()->getInitModel()->getOutputDim(), 8);
  EXPECT_EQ(dynamics.getHelper()->getInitModel()->getHiddenDim(), 20);
  EXPECT_EQ(dynamics.getHelper()->getInitModel()->getOutputModel()->getInputDim(), 23);
  EXPECT_EQ(dynamics.getHelper()->getInitModel()->getOutputModel()->getOutputDim(), 8);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->getInputDim(), 4);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->getOutputDim(), 1);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->getHiddenDim(), 4);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->getOutputModel()->getInputDim(), 8);
  EXPECT_EQ(dynamics.getHelper()->getLSTMModel()->getOutputModel()->getOutputDim(), 1);
  EXPECT_EQ(dynamics.getHelper()->getInitLen(), 11);

  EXPECT_EQ(dynamics.getMeanHelper()->getInitModel()->getInputDim(), 9);
  EXPECT_EQ(dynamics.getMeanHelper()->getInitModel()->getOutputDim(), 8);
  EXPECT_EQ(dynamics.getMeanHelper()->getInitModel()->getHiddenDim(), 20);
  EXPECT_EQ(dynamics.getMeanHelper()->getInitModel()->getOutputModel()->getInputDim(), 29);
  EXPECT_EQ(dynamics.getMeanHelper()->getInitModel()->getOutputModel()->getOutputDim(), 8);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->getInputDim(), 11);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->getOutputDim(), 2);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->getHiddenDim(), 4);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->getOutputModel()->getInputDim(), 15);
  EXPECT_EQ(dynamics.getMeanHelper()->getLSTMModel()->getOutputModel()->getOutputDim(), 2);
  EXPECT_EQ(dynamics.getMeanHelper()->getInitLen(), 11);

  EXPECT_EQ(dynamics.getUncertaintyHelper()->getInitModel()->getInputDim(), 10);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getInitModel()->getOutputDim(), 8);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getInitModel()->getHiddenDim(), 20);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getInitModel()->getOutputModel()->getInputDim(), 30);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getInitModel()->getOutputModel()->getOutputDim(), 8);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->getInputDim(), 12);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->getOutputDim(), 7);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->getHiddenDim(), 4);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->getOutputModel()->getInputDim(), 16);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getLSTMModel()->getOutputModel()->getOutputDim(), 7);
  EXPECT_EQ(dynamics.getUncertaintyHelper()->getInitLen(), 11);
}

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestGPUvsCPU)
{
  CudaCheckError();
  using DYN = RacerDubinsElevationLSTMUncertainty;
  RacerDubinsElevationLSTMUncertainty dynamics =
      RacerDubinsElevationLSTMUncertainty(mppi::tests::racer_dubins_elevation_uncertainty_test);

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);

  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
  TwoDTextureHelper<float4>* normal_helper = dynamics.getTextureHelperNormals();
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, origin);
  helper->updateTexture(0, data_vec, extent);
  helper->updateResolution(0, map_resolution);
  helper->enableTexture(0);
  helper->copyToDevice(false);
  normal_helper->updateRotation(0, new_rot_mat);
  normal_helper->updateOrigin(0, origin);
  normal_helper->updateTexture(0, normal_vec, extent);
  normal_helper->updateResolution(0, map_resolution);
  normal_helper->enableTexture(0);
  normal_helper->copyToDevice(true);

  DYN::buffer_trajectory buffer;
  buffer["VEL_X"] = Eigen::VectorXf::Random(51);
  buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
  buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
  buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);
  buffer["ROLL"] = Eigen::VectorXf::Random(51);
  buffer["PITCH"] = Eigen::VectorXf::Random(51);
  buffer["THROTTLE_CMD"] = Eigen::VectorXf::Random(51);
  buffer["BRAKE_STATE"] = Eigen::VectorXf::Random(51);
  buffer["BRAKE_CMD"] = Eigen::VectorXf::Random(51);

  checkGPUComputationStep<RacerDubinsElevationLSTMUncertainty>(dynamics, 0.02f, 16, 32, buffer, 1.0e-4);
}

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestStepGPUvsCPUReverse)
{
  using DYN = RacerDubinsElevationLSTMUncertainty;

  CudaCheckError();
  using DYN = RacerDubinsElevationLSTMUncertainty;
  RacerDubinsElevationLSTMUncertainty dynamics =
      RacerDubinsElevationLSTMUncertainty(mppi::tests::racer_dubins_elevation_uncertainty_test);
  auto params = dynamics.getParams();
  params.gear_sign = -1;
  dynamics.setParams(params);

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);

  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
  TwoDTextureHelper<float4>* normal_helper = dynamics.getTextureHelperNormals();
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, origin);
  helper->updateTexture(0, data_vec, extent);
  helper->updateResolution(0, map_resolution);
  helper->enableTexture(0);
  helper->copyToDevice(false);
  normal_helper->updateRotation(0, new_rot_mat);
  normal_helper->updateOrigin(0, origin);
  normal_helper->updateTexture(0, normal_vec, extent);
  normal_helper->updateResolution(0, map_resolution);
  normal_helper->enableTexture(0);
  normal_helper->copyToDevice(true);

  DYN::buffer_trajectory buffer;
  buffer["VEL_X"] = Eigen::VectorXf::Random(51);
  buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
  buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
  buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);
  buffer["ROLL"] = Eigen::VectorXf::Random(51);
  buffer["PITCH"] = Eigen::VectorXf::Random(51);
  buffer["THROTTLE_CMD"] = Eigen::VectorXf::Random(51);
  buffer["BRAKE_STATE"] = Eigen::VectorXf::Random(51);
  buffer["BRAKE_CMD"] = Eigen::VectorXf::Random(51);

  checkGPUComputationStep<RacerDubinsElevationLSTMUncertainty>(dynamics, 0.02f, 16, 32, buffer, 1.0e-4);
}