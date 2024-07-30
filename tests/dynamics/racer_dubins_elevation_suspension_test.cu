#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/racer_dubins/racer_dubins_elevation_suspension_lstm.cuh>
#include <kernel_tests/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <racer_test_networks.h>
#include <cuda_runtime.h>

class RacerDubinsElevationSuspensionTest : public ::testing::Test
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
};

TEST_F(RacerDubinsElevationSuspensionTest, Template)
{
  using DYNAMICS = RacerDubinsElevationSuspension;
  DYNAMICS dynamics = DYNAMICS(mppi::tests::steering_lstm, stream);
  EXPECT_EQ(24, DYNAMICS::STATE_DIM);
  EXPECT_EQ(2, DYNAMICS::CONTROL_DIM);
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
}

TEST_F(RacerDubinsElevationSuspensionTest, TestStepGPUvsCPU)
{
  const int num_rollouts = 1000;
  const float dt = 0.1f;
  CudaCheckError();
  using DYN = RacerDubinsElevationSuspension;
  DYN dynamics = DYN(mppi::tests::steering_lstm, stream);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 17.590296);
  EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 3.286375);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_constant, 9.301527);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_drag_constant, -0.60327667);

  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
  helper->setExtent(0, extent);

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, origin);

  helper->updateTexture(0, data_vec);
  helper->updateResolution(0, map_resolution);
  helper->enableTexture(0);
  helper->copyToDevice(true);

  CudaCheckError();
  dynamics.GPUSetup();
  CudaCheckError();

  EXPECT_NE(dynamics.getHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.network_d_, nullptr);
  EXPECT_EQ(dynamics.network_d_, dynamics.getHelper()->getLSTMDevicePtr());

  Eigen::Matrix<float, DYN::CONTROL_DIM, num_rollouts> control_trajectory;
  control_trajectory = Eigen::Matrix<float, DYN::CONTROL_DIM, num_rollouts>::Random();
  Eigen::Matrix<float, DYN::STATE_DIM, num_rollouts> state_trajectory;
  state_trajectory = Eigen::Matrix<float, DYN::STATE_DIM, num_rollouts>::Random();

  std::vector<std::array<float, DYN::STATE_DIM>> s(num_rollouts);
  std::vector<std::array<float, DYN::STATE_DIM>> s_next(num_rollouts);
  std::vector<std::array<float, DYN::STATE_DIM>> s_der(num_rollouts);
  // steering, throttle
  std::vector<std::array<float, DYN::CONTROL_DIM>> u(num_rollouts);

  DYN::state_array state;
  DYN::state_array next_state_cpu;
  DYN::control_array control;
  DYN::output_array output;
  DYN::state_array state_der_cpu = DYN::state_array::Zero();

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for (int y_dim = 1; y_dim <= 16; y_dim++)
  {
    DYN::buffer_trajectory buffer;
    buffer["VEL_X"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);

    for (int state_index = 0; state_index < num_rollouts; state_index++)
    {
      for (int dim = 0; dim < s[0].size(); dim++)
      {
        s[state_index][dim] = state_trajectory.col(state_index)(dim);
      }
      for (int dim = 0; dim < u[0].size(); dim++)
      {
        u[state_index][dim] = control_trajectory.col(state_index)(dim);
      }
    }
    dynamics.updateFromBuffer(buffer);
    launchStepTestKernel<DYN, 16>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
    for (int point = 0; point < num_rollouts; point++)
    {
      dynamics.initializeDynamics(state, control, output, 0, 0);
      state = state_trajectory.col(point);
      control = control_trajectory.col(point);
      state_der_cpu = DYN::state_array::Zero();

      dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
      // for (int dim = 0; dim < DYN::STATE_DIM; dim++)
      for (int dim = 0; dim < DYN::STATE_DIM; dim++)
      {
        EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        // EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_NEAR(next_state_cpu(dim), s_next[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        EXPECT_TRUE(isfinite(s_next[point][dim]));
      }
    }
  }
  dynamics.freeCudaMem();
}

TEST_F(RacerDubinsElevationSuspensionTest, TestStepGPUvsCPUWithPartialElevationMap)
{
  const int num_rollouts = 1000;
  const float dt = 0.1f;
  CudaCheckError();
  using DYN = RacerDubinsElevationSuspension;
  DYN dynamics = DYN(mppi::tests::steering_lstm, stream);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 17.590296);
  EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 3.286375);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_constant, 9.301527);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_drag_constant, -0.60327667);

  for (int i = 0; i < data_vec.size(); i++)
  {
    if (i < 10)
    {
      data_vec[i] = NAN;
    }
  }
  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, origin);
  helper->updateTexture(0, data_vec, extent);
  helper->updateResolution(0, map_resolution);
  helper->enableTexture(0);
  helper->copyToDevice(true);

  CudaCheckError();
  dynamics.GPUSetup();
  CudaCheckError();

  EXPECT_NE(dynamics.getHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.network_d_, nullptr);
  EXPECT_EQ(dynamics.network_d_, dynamics.getHelper()->getLSTMDevicePtr());

  Eigen::Matrix<float, DYN::CONTROL_DIM, num_rollouts> control_trajectory;
  control_trajectory = Eigen::Matrix<float, DYN::CONTROL_DIM, num_rollouts>::Random();
  Eigen::Matrix<float, DYN::STATE_DIM, num_rollouts> state_trajectory;
  state_trajectory = Eigen::Matrix<float, DYN::STATE_DIM, num_rollouts>::Random();

  std::vector<std::array<float, DYN::STATE_DIM>> s(num_rollouts);
  std::vector<std::array<float, DYN::STATE_DIM>> s_next(num_rollouts);
  std::vector<std::array<float, DYN::STATE_DIM>> s_der(num_rollouts);
  // steering, throttle
  std::vector<std::array<float, DYN::CONTROL_DIM>> u(num_rollouts);

  DYN::state_array state;
  DYN::state_array next_state_cpu;
  DYN::control_array control;
  DYN::output_array output;
  DYN::state_array state_der_cpu = DYN::state_array::Zero();

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for (int y_dim = 1; y_dim <= 16; y_dim++)
  {
    DYN::buffer_trajectory buffer;
    buffer["VEL_X"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);

    for (int state_index = 0; state_index < num_rollouts; state_index++)
    {
      for (int dim = 0; dim < s[0].size(); dim++)
      {
        s[state_index][dim] = state_trajectory.col(state_index)(dim);
      }
      for (int dim = 0; dim < u[0].size(); dim++)
      {
        u[state_index][dim] = control_trajectory.col(state_index)(dim);
      }
    }
    dynamics.updateFromBuffer(buffer);
    launchStepTestKernel<DYN, 16>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
    for (int point = 0; point < num_rollouts; point++)
    {
      dynamics.initializeDynamics(state, control, output, 0, 0);
      state = state_trajectory.col(point);
      control = control_trajectory.col(point);
      state_der_cpu = DYN::state_array::Zero();

      dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
      // for (int dim = 0; dim < DYN::STATE_DIM; dim++)
      for (int dim = 0; dim < DYN::STATE_DIM; dim++)
      {
        EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        // EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_NEAR(next_state_cpu(dim), s_next[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        EXPECT_TRUE(isfinite(s_next[point][dim]));
      }
    }
  }
  dynamics.freeCudaMem();
}

TEST_F(RacerDubinsElevationSuspensionTest, TestStepGPUvsCPUWithPartialElevationAndNormalMap)
{
  const int num_rollouts = 1000;
  const float dt = 0.1f;
  CudaCheckError();
  using DYN = RacerDubinsElevationSuspension;
  DYN dynamics = DYN(mppi::tests::steering_lstm, stream);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 17.590296);
  EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 3.286375);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_constant, 9.301527);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_drag_constant, -0.60327667);

  for (int i = 0; i < data_vec.size(); i++)
  {
    if (i < 10)
    {
      data_vec[i] = NAN;
      normal_vec[i] = make_float4(NAN, NAN, NAN, NAN);
    }
  }

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

  CudaCheckError();
  dynamics.GPUSetup();
  CudaCheckError();

  EXPECT_NE(dynamics.getHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.network_d_, nullptr);
  EXPECT_EQ(dynamics.network_d_, dynamics.getHelper()->getLSTMDevicePtr());

  Eigen::Matrix<float, DYN::CONTROL_DIM, num_rollouts> control_trajectory;
  control_trajectory = Eigen::Matrix<float, DYN::CONTROL_DIM, num_rollouts>::Random();
  Eigen::Matrix<float, DYN::STATE_DIM, num_rollouts> state_trajectory;
  state_trajectory = Eigen::Matrix<float, DYN::STATE_DIM, num_rollouts>::Random();

  std::vector<std::array<float, DYN::STATE_DIM>> s(num_rollouts);
  std::vector<std::array<float, DYN::STATE_DIM>> s_next(num_rollouts);
  std::vector<std::array<float, DYN::STATE_DIM>> s_der(num_rollouts);
  // steering, throttle
  std::vector<std::array<float, DYN::CONTROL_DIM>> u(num_rollouts);

  DYN::state_array state;
  DYN::state_array next_state_cpu;
  DYN::control_array control;
  DYN::output_array output;
  DYN::state_array state_der_cpu = DYN::state_array::Zero();

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for (int y_dim = 1; y_dim <= 16; y_dim++)
  {
    DYN::buffer_trajectory buffer;
    buffer["VEL_X"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);

    for (int state_index = 0; state_index < num_rollouts; state_index++)
    {
      for (int dim = 0; dim < s[0].size(); dim++)
      {
        s[state_index][dim] = state_trajectory.col(state_index)(dim);
      }
      for (int dim = 0; dim < u[0].size(); dim++)
      {
        u[state_index][dim] = control_trajectory.col(state_index)(dim);
      }
    }
    dynamics.updateFromBuffer(buffer);
    launchStepTestKernel<DYN, 16>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
    for (int point = 0; point < num_rollouts; point++)
    {
      dynamics.initializeDynamics(state, control, output, 0, 0);
      state = state_trajectory.col(point);
      control = control_trajectory.col(point);
      state_der_cpu = DYN::state_array::Zero();

      dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
      // for (int dim = 0; dim < DYN::STATE_DIM; dim++)
      for (int dim = 0; dim < DYN::STATE_DIM; dim++)
      {
        EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        // EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_NEAR(next_state_cpu(dim), s_next[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        EXPECT_TRUE(isfinite(s_next[point][dim]));
      }
    }
  }
  dynamics.freeCudaMem();
}
