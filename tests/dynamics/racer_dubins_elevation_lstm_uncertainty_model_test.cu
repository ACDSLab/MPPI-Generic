#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/racer_dubins/racer_dubins_elevation_lstm_unc.cuh>
#include <kernel_tests/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <racer_test_networks.h>
#include <cuda_runtime.h>

class RacerDubinsElevationLSTMUncertaintyTest : public ::testing::Test
{
public:
  cudaStream_t stream;

  void SetUp() override
  {
    CudaCheckError();
    HANDLE_ERROR(cudaStreamCreate(&stream));
  }

  void TearDown() override
  {
    CudaCheckError();
    HANDLE_ERROR(cudaStreamDestroy(stream));
  }
};

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, Template)
{
  auto dynamics = RacerDubinsElevationLSTMUncertainty();
  EXPECT_EQ(19, RacerDubinsElevationLSTMUncertainty::STATE_DIM);
  EXPECT_EQ(2, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM);
  EXPECT_TRUE(dynamics.checkRequiresBuffer());
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  // const int blk = RacerDubinsElevationLSTMUncertainty::SHARED_MEM_REQUEST_BLK_BYTES;
  // EXPECT_EQ(blk, 324);
  // const int grd = RacerDubinsElevationLSTMUncertainty::SHARED_MEM_REQUEST_GRD_BYTES;
  // EXPECT_EQ(grd, 0);
}

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, BindStream)
{
  auto dynamics = RacerDubinsElevationLSTMUncertainty(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);
}

TEST_F(RacerDubinsElevationLSTMUncertaintyTest, loadNN)
{
  auto dynamics = RacerDubinsElevationLSTMUncertainty(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);
}

// TODO test case on loading the NN's and making sure the outputs look good
//
// /*
// float c_t = 1.3;
// float c_b = 2.5;
// float c_v = 3.7;
// float c_0 = 4.9;
// float wheel_base = 0.3;
//  */
//
// // TEST_F(RacerDubinsElevationLSTMUncertaintyTest, ComputeDynamics)
// // {
// //   RacerDubinsElevationLSTMUncertainty dynamics = RacerDubinsElevationLSTMUncertainty();
// //   auto params = dynamics.getParams();
// //   RacerDubinsElevationLSTMUncertainty::state_array x = RacerDubinsElevationLSTMUncertainty::state_array::Zero();
// //   RacerDubinsElevationLSTMUncertainty::control_array u =
// RacerDubinsElevationLSTMUncertainty::control_array::Zero();
//
// //   // computeDynamics should not touch the roll/pitch element
// //   RacerDubinsElevationLSTMUncertainty::state_array next_x =
// RacerDubinsElevationLSTMUncertainty::state_array::Ones() *
// //   0.153; dynamics.computeDynamics(x, u, next_x); EXPECT_FLOAT_EQ(next_x(0), 4.9); EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_FLOAT_EQ(next_x(2), 0);
// //   EXPECT_FLOAT_EQ(next_x(3), 0);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << 1, M_PI_2, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << 1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 2.6 - 4.7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_NEAR(next_x(2), 0, 1e-7);
// //   EXPECT_FLOAT_EQ(next_x(3), 1);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << 1, M_PI_2, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << -1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.5 - 4.7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_NEAR(next_x(2), 0, 1e-7);
// //   EXPECT_FLOAT_EQ(next_x(3), 1);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << -1, 0, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << 1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 4.7 + 2.6 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_FLOAT_EQ(next_x(2), -1);
// //   EXPECT_FLOAT_EQ(next_x(3), 0);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << -1, 0, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << -1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 3.5 + 4.7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_FLOAT_EQ(next_x(2), -1);
// //   EXPECT_FLOAT_EQ(next_x(3), 0);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << 7, 0, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << 1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 3.9 - 5.7 * 7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_FLOAT_EQ(next_x(2), 7);
// //   EXPECT_FLOAT_EQ(next_x(3), 0);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << -7, 0, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << 1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 3.9 + 5.7 * 7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_FLOAT_EQ(next_x(2), -7);
// //   EXPECT_FLOAT_EQ(next_x(3), 0);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << 7, 0, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << -1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 4.5 - 5.7 * 7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_FLOAT_EQ(next_x(2), 7);
// //   EXPECT_FLOAT_EQ(next_x(3), 0);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << -7, 0, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << -1, 0;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 4.5 + 5.7 * 7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), 0);
// //   EXPECT_FLOAT_EQ(next_x(2), -7);
// //   EXPECT_FLOAT_EQ(next_x(3), 0);
// //   EXPECT_FLOAT_EQ(next_x(4), 0);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << 1, M_PI_2, 0, 3, 0, 0.5, -0.5, 0.0, 0.0;
// //   u << 0, 1;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 4.7 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), (1 / .3) * tan(0));
// //   EXPECT_NEAR(next_x(2), 0, 1e-7);
// //   EXPECT_FLOAT_EQ(next_x(3), 1);
// //   EXPECT_FLOAT_EQ(next_x(4), 5 * 0.6);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << -1, M_PI_2, 0, 3, 5.0, 0.5, -0.5, 0.0, 0.0;
// //   u << -1, -1;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 4.7 + 3.5 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), (-1 / .3) * tan(5.0 / -10.2));
// //   EXPECT_NEAR(next_x(2), 0, 1e-7);
// //   EXPECT_FLOAT_EQ(next_x(3), -1);
// //   EXPECT_FLOAT_EQ(next_x(4), -5);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << -0.4, M_PI_2, 0, 3, 5.0, 0.5, -0.5, 0.0, 0.0;
// //   u << -1, -1;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 3.7 * 0.4 + 2.5 * 0.4 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), (-0.4 / .3) * tan(5.0 / -9.1));
// //   EXPECT_NEAR(next_x(2), 0, 1e-7);
// //   EXPECT_FLOAT_EQ(next_x(3), -0.4);
// //   EXPECT_FLOAT_EQ(next_x(4), -5);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
// //   x << 0.4, M_PI_2, 0, 3, 5.0, 0.5, -0.5, 0.0, 0.0;
// //   u << 0.1, -1;
// //   dynamics.computeDynamics(x, u, next_x);
// //   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.7 * 0.4 - sinf(-0.5) * -9.81);
// //   EXPECT_FLOAT_EQ(next_x(1), (0.4 / .3) * tan(5.0 / -9.1));
// //   EXPECT_NEAR(next_x(2), 0, 1e-7);
// //   EXPECT_FLOAT_EQ(next_x(3), 0.4);
// //   EXPECT_FLOAT_EQ(next_x(4), -5);
// //   EXPECT_FLOAT_EQ(next_x(5), 0.153);
// //   EXPECT_FLOAT_EQ(next_x(6), 0.153);
// // }
//
// // TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestModelGPU)
// // {
// //   RacerDubinsElevationLSTMUncertainty dynamics = RacerDubinsElevationLSTMUncertainty();
// //   dynamics.GPUSetup();
// //   CudaCheckError();
//
// //   Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM, 100> control_trajectory;
// //   control_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM, 100>::Random();
// //   Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM, 100> state_trajectory;
// //   state_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM, 100>::Random();
//
// //   std::vector<std::array<float, 9>> s(100);
// //   std::vector<std::array<float, 9>> s_der(100);
// //   // steering, throttle
// //   std::vector<std::array<float, 2>> u(100);
// //   for (int state_index = 0; state_index < s.size(); state_index++)
// //   {
// //     for (int dim = 0; dim < s[0].size(); dim++)
// //     {
// //       s[state_index][dim] = state_trajectory.col(state_index)(dim);
// //     }
// //     for (int dim = 0; dim < u[0].size(); dim++)
// //     {
// //       u[state_index][dim] = control_trajectory.col(state_index)(dim);
// //     }
// //   }
//
// //   // These variables will be changed so initialized to the right size only
//
// //   // Run dynamics on dynamicsU
// //   // Run dynamics on GPU
// //   for (int y_dim = 1; y_dim <= 4; y_dim++)
// //   {
// //     launchComputeDynamicsTestKernel<RacerDubinsElevationLSTMUncertainty, 9, 2>(dynamics, s, u, s_der, y_dim);
// //     for (int point = 0; point < 100; point++)
// //     {
// //       RacerDubinsElevationLSTMUncertainty::state_array state = state_trajectory.col(point);
// //       RacerDubinsElevationLSTMUncertainty::control_array control = control_trajectory.col(point);
// //       RacerDubinsElevationLSTMUncertainty::state_array state_der_cpu =
// //       RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
// //       dynamics.computeDynamics(state, control, state_der_cpu);
// //       for (int dim = 0; dim < 6; dim++)
// //       {
// //         EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-5)
// //             << "at point " << point << " dim " << dim << " with y_dim " << y_dim;
// //         EXPECT_TRUE(isfinite(s_der[point][dim]));
// //       }
// //     }
// //   }
//
// //   dynamics.freeCudaMem();
// //   CudaCheckError();
// // }
//
// // TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestUpdateState)
// // {
// //   CudaCheckError();
// //   RacerDubinsElevationLSTMUncertainty dynamics = RacerDubinsElevationLSTMUncertainty();
// //   RacerDubinsElevationLSTMUncertainty::state_array state;
// //   RacerDubinsElevationLSTMUncertainty::state_array next_state;
// //   RacerDubinsElevationLSTMUncertainty::state_array state_der;
//
// //   // TODO add in the elevation map
//
// //   state << 0, 0, 0, 0, 0, -0.5, 0.5;
// //   state_der << 1, 1, 1, 1, 1, 0, 0;
// //   dynamics.updateState(state, next_state, state_der, 0.1);
// //   EXPECT_TRUE(state_der != RacerDubinsElevationLSTMUncertainty::state_array::Zero());
// //   EXPECT_FLOAT_EQ(next_state(0), 0.1);
// //   EXPECT_FLOAT_EQ(next_state(1), 0.1);
// //   EXPECT_FLOAT_EQ(next_state(2), 0.1);
// //   EXPECT_FLOAT_EQ(next_state(3), 0.1);
// //   EXPECT_FLOAT_EQ(next_state(4), 0.1);
// //   EXPECT_FLOAT_EQ(next_state(5), 0.0);
// //   EXPECT_FLOAT_EQ(next_state(6), 0.0);
//
// //   state << 0, M_PI - 0.1, 0, 0, 0, -0.5, 0.5;
// //   state_der << 1, 1, 1, 1, 1;
// //   dynamics.updateState(state, next_state, state_der, 1.0);
// //   EXPECT_TRUE(state_der != RacerDubinsElevationLSTMUncertainty::state_array::Zero());
// //   EXPECT_FLOAT_EQ(next_state(0), 1.0);
// //   EXPECT_FLOAT_EQ(next_state(1), 1.0 - M_PI - 0.1);
// //   EXPECT_FLOAT_EQ(next_state(2), 1.0);
// //   EXPECT_FLOAT_EQ(next_state(3), 1.0);
// //   EXPECT_FLOAT_EQ(next_state(4), 0.5);
// //   EXPECT_FLOAT_EQ(next_state(5), 0.0);
// //   EXPECT_FLOAT_EQ(next_state(6), 0.0);
//
// //   state << 0, -M_PI + 0.1, 0, 0, 0, -0.5, 0.5;
// //   state_der << 1, -1, 1, 1, 1;
// //   dynamics.updateState(state, next_state, state_der, 1.0);
// //   EXPECT_TRUE(state_der != RacerDubinsElevationLSTMUncertainty::state_array::Zero());
// //   EXPECT_FLOAT_EQ(next_state(0), 1.0);
// //   EXPECT_FLOAT_EQ(next_state(1), M_PI + 0.1 - 1.0);
// //   EXPECT_FLOAT_EQ(next_state(2), 1.0);
// //   EXPECT_FLOAT_EQ(next_state(3), 1.0);
// //   EXPECT_FLOAT_EQ(next_state(4), 0.5);
// //   EXPECT_FLOAT_EQ(next_state(5), 0.0);
// //   EXPECT_FLOAT_EQ(next_state(6), 0.0);
//
// //   CudaCheckError();
// // }
//
// TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestStepGPUvsCPU)
// {
//   const int num_rollouts = 1000;
//   const float dt = 0.1f;
//   CudaCheckError();
//   using DYN = RacerDubinsElevationLSTMUncertainty;
//   RacerDubinsElevationLSTMUncertainty dynamics = RacerDubinsElevationLSTMUncertainty(mppi::tests::steering_lstm);
//   EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 17.590296);
//   EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 3.286375);
//   EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_constant, 9.301527);
//   EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_drag_constant, -0.60327667);
//
//   cudaExtent extent = make_cudaExtent(10, 20, 0);
//   TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
//   helper->setExtent(0, extent);
//
//   std::vector<float> data_vec;
//   data_vec.resize(10 * 20);
//   for (int i = 0; i < data_vec.size(); i++)
//   {
//     data_vec[i] = i * 1.0f;
//   }
//
//   std::array<float3, 3> new_rot_mat{};
//   new_rot_mat[0] = make_float3(0, 1, 0);
//   new_rot_mat[1] = make_float3(1, 0, 0);
//   new_rot_mat[2] = make_float3(0, 0, 1);
//   helper->updateRotation(0, new_rot_mat);
//   helper->updateOrigin(0, make_float3(1, 2, 3));
//
//   helper->updateTexture(0, data_vec);
//   helper->updateResolution(0, 10);
//   helper->enableTexture(0);
//   helper->copyToDevice(true);
//
//   CudaCheckError();
//   dynamics.GPUSetup();
//   CudaCheckError();
//
//   EXPECT_NE(dynamics.getHelper()->getLSTMDevicePtr(), nullptr);
//   EXPECT_NE(dynamics.network_d_, nullptr);
//   EXPECT_EQ(dynamics.network_d_, dynamics.getHelper()->getLSTMDevicePtr());
//
//   Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM, num_rollouts> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM,
//   num_rollouts>::Random(); Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM, num_rollouts>
//   state_trajectory; state_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM,
//   num_rollouts>::Random();
//
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM>> s(num_rollouts);
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM>> s_next(num_rollouts);
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM>> s_der(num_rollouts);
//   // steering, throttle
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM>> u(num_rollouts);
//
//   RacerDubinsElevationLSTMUncertainty::state_array state;
//   RacerDubinsElevationLSTMUncertainty::state_array next_state_cpu;
//   RacerDubinsElevationLSTMUncertainty::control_array control;
//   RacerDubinsElevationLSTMUncertainty::output_array output;
//   RacerDubinsElevationLSTMUncertainty::state_array state_der_cpu =
//   RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//   // Run dynamics on dynamicsU
//   // Run dynamics on GPU
//   for (int y_dim = 1; y_dim <= 16; y_dim++)
//   {
//     DYN::buffer_trajectory buffer;
//     buffer["VEL_X"] = Eigen::VectorXf::Random(51);
//     buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
//     buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
//     buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);
//
//     for (int state_index = 0; state_index < num_rollouts; state_index++)
//     {
//       for (int dim = 0; dim < s[0].size(); dim++)
//       {
//         s[state_index][dim] = state_trajectory.col(state_index)(dim);
//       }
//       for (int dim = 0; dim < u[0].size(); dim++)
//       {
//         u[state_index][dim] = control_trajectory.col(state_index)(dim);
//       }
//     }
//     dynamics.updateFromBuffer(buffer);
//     launchStepTestKernel<RacerDubinsElevationLSTMUncertainty, 16>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
//     for (int point = 0; point < num_rollouts; point++)
//     {
//       dynamics.initializeDynamics(state, control, output, 0, 0);
//       state = state_trajectory.col(point);
//       control = control_trajectory.col(point);
//       state_der_cpu = RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//       dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//       // for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::STATE_DIM; dim++)
//       for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::STATE_DIM; dim++)
//       {
//         EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-4)
//             << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
//         // EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//         EXPECT_NEAR(next_state_cpu(dim), s_next[point][dim], 1e-4)
//             << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
//         EXPECT_TRUE(isfinite(s_next[point][dim]));
//       }
//     }
//   }
//   dynamics.freeCudaMem();
// }
//
// TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestStepGPUvsCPUReverse)
// {
//   using DYN = RacerDubinsElevationLSTMUncertainty;
//
//   const int num_rollouts = 1000;
//   const float dt = 0.1f;
//   CudaCheckError();
//   RacerDubinsElevationLSTMUncertainty dynamics = RacerDubinsElevationLSTMUncertainty(mppi::tests::steering_lstm);
//   auto params = dynamics.getParams();
//   params.gear_sign = -1;
//   dynamics.setParams(params);
//   EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 17.590296);
//   EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 3.286375);
//   EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_constant, 9.301527);
//   EXPECT_FLOAT_EQ(dynamics.getParams().steer_accel_drag_constant, -0.60327667);
//
//   cudaExtent extent = make_cudaExtent(10, 20, 0);
//   TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
//   helper->setExtent(0, extent);
//
//   std::vector<float> data_vec;
//   data_vec.resize(10 * 20);
//   for (int i = 0; i < data_vec.size(); i++)
//   {
//     data_vec[i] = i * 1.0f;
//   }
//
//   std::array<float3, 3> new_rot_mat{};
//   new_rot_mat[0] = make_float3(0, 1, 0);
//   new_rot_mat[1] = make_float3(1, 0, 0);
//   new_rot_mat[2] = make_float3(0, 0, 1);
//   helper->updateRotation(0, new_rot_mat);
//   helper->updateOrigin(0, make_float3(1, 2, 3));
//
//   helper->updateTexture(0, data_vec);
//   helper->updateResolution(0, 10);
//   helper->enableTexture(0);
//   helper->copyToDevice(true);
//
//   CudaCheckError();
//   dynamics.GPUSetup();
//   CudaCheckError();
//
//   EXPECT_NE(dynamics.getHelper()->getLSTMDevicePtr(), nullptr);
//   EXPECT_NE(dynamics.network_d_, nullptr);
//   EXPECT_EQ(dynamics.network_d_, dynamics.getHelper()->getLSTMDevicePtr());
//
//   Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM, num_rollouts> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM,
//   num_rollouts>::Random(); Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM, num_rollouts>
//   state_trajectory; state_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM,
//   num_rollouts>::Random();
//
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM>> s(num_rollouts);
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM>> s_next(num_rollouts);
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM>> s_der(num_rollouts);
//   // steering, throttle
//   std::vector<std::array<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM>> u(num_rollouts);
//
//   RacerDubinsElevationLSTMUncertainty::state_array state;
//   RacerDubinsElevationLSTMUncertainty::state_array next_state_cpu;
//   RacerDubinsElevationLSTMUncertainty::control_array control;
//   RacerDubinsElevationLSTMUncertainty::output_array output;
//   RacerDubinsElevationLSTMUncertainty::state_array state_der_cpu =
//   RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//   // Run dynamics on dynamicsU
//   // Run dynamics on GPU
//   for (int y_dim = 1; y_dim <= 16; y_dim++)
//   {
//     DYN::buffer_trajectory buffer;
//     buffer["VEL_X"] = Eigen::VectorXf::Random(51);
//     buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
//     buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
//     buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);
//
//     for (int state_index = 0; state_index < num_rollouts; state_index++)
//     {
//       for (int dim = 0; dim < s[0].size(); dim++)
//       {
//         s[state_index][dim] = state_trajectory.col(state_index)(dim);
//       }
//       for (int dim = 0; dim < u[0].size(); dim++)
//       {
//         u[state_index][dim] = control_trajectory.col(state_index)(dim);
//       }
//     }
//     dynamics.updateFromBuffer(buffer);
//     launchStepTestKernel<RacerDubinsElevationLSTMUncertainty, 16>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
//     for (int point = 0; point < num_rollouts; point++)
//     {
//       dynamics.initializeDynamics(state, control, output, 0, 0);
//       state = state_trajectory.col(point);
//       control = control_trajectory.col(point);
//       state_der_cpu = RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//       dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//       // for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::STATE_DIM; dim++)
//       for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::STATE_DIM; dim++)
//       {
//         EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-4)
//             << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
//         // EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//         EXPECT_NEAR(next_state_cpu(dim), s_next[point][dim], 1e-4)
//             << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
//         EXPECT_TRUE(isfinite(s_next[point][dim]));
//       }
//     }
//   }
//   dynamics.freeCudaMem();
// }
//
// TEST_F(RacerDubinsElevationLSTMUncertaintyTest, compareToElevationWithoutUncertainty)
// {
//   // by default the network will output zeros and not effect any states
//   using DYN = RacerDubinsElevationLSTMUncertainty;
//
//   const int num_rollouts = 1000;
//   const float dt = 0.1f;
//   CudaCheckError();
//   RacerDubinsElevationLSTMUncertainty dynamics = RacerDubinsElevationLSTMUncertainty();
//   RacerDubinsElevation dynamics2 = RacerDubinsElevation();
//   auto params = dynamics.getParams();
//   dynamics.setParams(params);
//   dynamics2.setParams(params);
//
//   cudaExtent extent = make_cudaExtent(10, 20, 0);
//   TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
//   helper->setExtent(0, extent);
//
//   std::vector<float> data_vec;
//   data_vec.resize(10 * 20);
//   for (int i = 0; i < data_vec.size(); i++)
//   {
//     data_vec[i] = i * 1.0f;
//   }
//
//   std::array<float3, 3> new_rot_mat{};
//   new_rot_mat[0] = make_float3(0, 1, 0);
//   new_rot_mat[1] = make_float3(1, 0, 0);
//   new_rot_mat[2] = make_float3(0, 0, 1);
//   helper->updateRotation(0, new_rot_mat);
//   helper->updateOrigin(0, make_float3(1, 2, 3));
//
//   helper->updateTexture(0, data_vec);
//   helper->updateResolution(0, 10);
//   helper->enableTexture(0);
//   helper->copyToDevice(true);
//
//   TwoDTextureHelper<float>* helper2 = dynamics2.getTextureHelper();
//   helper2->setExtent(0, extent);
//
//   helper2->updateRotation(0, new_rot_mat);
//   helper2->updateOrigin(0, make_float3(1, 2, 3));
//
//   data_vec.resize(10 * 20);
//   for (int i = 0; i < data_vec.size(); i++)
//   {
//     data_vec[i] = i * 1.0f;
//   }
//   helper2->updateTexture(0, data_vec);
//   helper2->updateResolution(0, 10);
//   helper2->enableTexture(0);
//   helper2->copyToDevice(true);
//
//   CudaCheckError();
//   dynamics.GPUSetup();
//   dynamics2.GPUSetup();
//   CudaCheckError();
//
//   Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM, num_rollouts> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::CONTROL_DIM,
//   num_rollouts>::Random(); Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM, num_rollouts>
//   state_trajectory; state_trajectory = Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM,
//   num_rollouts>::Random();
//
//   RacerDubinsElevationLSTMUncertainty::state_array state;
//   RacerDubinsElevationLSTMUncertainty::state_array next_state_cpu;
//   RacerDubinsElevationLSTMUncertainty::control_array control;
//   RacerDubinsElevationLSTMUncertainty::output_array output;
//   RacerDubinsElevationLSTMUncertainty::state_array state_der_cpu =
//   RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//   RacerDubinsElevationLSTMUncertainty::state_array state2;
//   RacerDubinsElevationLSTMUncertainty::state_array next_state_cpu2;
//   RacerDubinsElevationLSTMUncertainty::control_array control2;
//   RacerDubinsElevationLSTMUncertainty::output_array output2;
//   RacerDubinsElevationLSTMUncertainty::state_array state_der_cpu2 =
//   RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//   DYN::buffer_trajectory buffer;
//   buffer["VEL_X"] = Eigen::VectorXf::Random(51);
//   buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
//   buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
//   buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);
//
//   dynamics.updateFromBuffer(buffer);
//   for (int point = 0; point < num_rollouts; point++)
//   {
//     dynamics.initializeDynamics(state, control, output, 0, 0);
//     state = state_trajectory.col(point);
//     control = control_trajectory.col(point);
//     state_der_cpu = RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//     dynamics2.initializeDynamics(state2, control2, output2, 0, 0);
//     state2 = state_trajectory.col(point);
//     control2 = control_trajectory.col(point);
//     state_der_cpu2 = RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//     dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//     dynamics2.step(state2, next_state_cpu2, state_der_cpu2, control2, output2, 0, dt);
//
//     for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::STATE_DIM; dim++)
//     {
//       if (dim == 4 or dim == 8)
//       {  // this is done since the steering wheel setup is different, accel version
//         continue;
//       }
//       EXPECT_NEAR(state_der_cpu(dim), state_der_cpu2(dim), 1e-4) << "state der at index " << point << " dim " << dim;
//       EXPECT_NEAR(next_state_cpu(dim), next_state_cpu2(dim), 1e-4) << "next state at index " << point << " dim " <<
//       dim;
//     }
//     for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::OUTPUT_DIM; dim++)
//     {
//       if (dim == 8 or dim == 9)
//       {  // this is done since the steering wheel setup is different, accel version
//         continue;
//       }
//       EXPECT_NEAR(output(dim), output2(dim), 1e-4) << "output at index " << point << " dim " << dim;
//     }
//   }
//
//   params.gear_sign = -1;
//   dynamics.setParams(params);
//   dynamics2.setParams(params);
//
//   // check in reverse as well
//   for (int point = 0; point < num_rollouts; point++)
//   {
//     dynamics.initializeDynamics(state, control, output, 0, 0);
//     state = state_trajectory.col(point);
//     control = control_trajectory.col(point);
//     state_der_cpu = RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//     dynamics2.initializeDynamics(state2, control2, output2, 0, 0);
//     state2 = state_trajectory.col(point);
//     control2 = control_trajectory.col(point);
//     state_der_cpu2 = RacerDubinsElevationLSTMUncertainty::state_array::Zero();
//
//     dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//     dynamics2.step(state2, next_state_cpu2, state_der_cpu2, control2, output2, 0, dt);
//
//     for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::STATE_DIM; dim++)
//     {
//       if (dim == 4 or dim == 8)
//       {  // this is done since the steering wheel setup is different, accel version
//         continue;
//       }
//       EXPECT_NEAR(state_der_cpu(dim), state_der_cpu2(dim), 1e-4) << "at index " << point << " dim " << dim;
//       EXPECT_NEAR(next_state_cpu(dim), next_state_cpu2(dim), 1e-4) << "at index " << point << " dim " << dim;
//     }
//     for (int dim = 0; dim < RacerDubinsElevationLSTMUncertainty::OUTPUT_DIM; dim++)
//     {
//       if (dim == 8 or dim == 9)
//       {  // this is done since the steering wheel setup is different, accel version
//         continue;
//       }
//       EXPECT_NEAR(output(dim), output2(dim), 1e-4) << "at index " << point << " dim " << dim;
//     }
//   }
//   dynamics.freeCudaMem();
// }
//
// /*
// class LinearDummy : public RacerDubinsElevationLSTMUncertainty {
// public:
//   bool computeGrad(const Eigen::Ref<const state_array> & state,
//                    const Eigen::Ref<const control_array>& control,
//                    Eigen::Ref<dfdx> A,
//                    Eigen::Ref<dfdu> B) {
//     return false;
//   };
// };
//
// TEST_F(RacerDubinsElevationLSTMUncertaintyTest, TestComputeGradComputation) {
//   Eigen::Matrix<float, RacerDubinsElevationLSTMUncertainty::STATE_DIM, RacerDubinsElevationLSTMUncertainty::STATE_DIM
//   +
// RacerDubinsElevationLSTMUncertainty::CONTROL_DIM> numeric_jac; Eigen::Matrix<float,
// RacerDubinsElevationLSTMUncertainty::STATE_DIM, RacerDubinsElevationLSTMUncertainty::STATE_DIM +
// RacerDubinsElevationLSTMUncertainty::CONTROL_DIM> analytic_jac; RacerDubinsElevationLSTMUncertainty::state_array
// state; state
// << 1, 2, 3, 4; RacerDubinsElevationLSTMUncertainty::control_array control; control << 5;
//
//   auto analytic_grad_model = RacerDubinsElevationLSTMUncertainty();
//
//   RacerDubinsElevationLSTMUncertainty::dfdx A_analytic = RacerDubinsElevationLSTMUncertainty::dfdx::Zero();
//   RacerDubinsElevationLSTMUncertainty::dfdu B_analytic = RacerDubinsElevationLSTMUncertainty::dfdu::Zero();
//
//   analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);
//
//   auto numerical_grad_model = LinearDummy();
//
//   std::shared_ptr<ModelWrapperDDP<LinearDummy>> ddp_model =
// std::make_shared<ModelWrapperDDP<LinearDummy>>(&numerical_grad_model);
//
//   analytic_jac.leftCols<RacerDubinsElevationLSTMUncertainty::STATE_DIM>() = A_analytic;
//   analytic_jac.rightCols<RacerDubinsElevationLSTMUncertainty::CONTROL_DIM>() = B_analytic;
//   numeric_jac = ddp_model->df(state, control);
//
//   ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n" << numeric_jac << "\nAnalytic
//   Jacobian\n"
// << analytic_jac;
// }
//
// */
