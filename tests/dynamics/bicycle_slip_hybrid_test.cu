#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "mppi/dynamics/bicycle_slip/bicycle_slip_hybrid.cuh"
#include <mppi/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <cuda_runtime.h>
#include <racer_test_networks.h>

class BicycleSlipHybridTest : public ::testing::Test
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

const double tol = 1e-5;

TEST_F(BicycleSlipHybridTest, Template)
{
  auto dynamics = BicycleSlipHybrid();
  EXPECT_EQ(12, BicycleSlipHybrid::STATE_DIM);
  EXPECT_EQ(2, BicycleSlipHybrid::CONTROL_DIM);
  EXPECT_TRUE(dynamics.checkRequiresBuffer());
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);

  EXPECT_NE(dynamics.getSteerHelper(), nullptr);
  EXPECT_NE(dynamics.getDelayHelper(), nullptr);
  EXPECT_NE(dynamics.getTerraHelper(), nullptr);
}

TEST_F(BicycleSlipHybridTest, BindStream)
{
  auto dynamics = BicycleSlipHybrid(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);

  EXPECT_NE(dynamics.getSteerHelper(), nullptr);
  EXPECT_EQ(dynamics.getSteerHelper()->getLSTMModel()->stream_, stream);
  EXPECT_NE(dynamics.getDelayHelper(), nullptr);
  EXPECT_EQ(dynamics.getDelayHelper()->getLSTMModel()->stream_, stream);
  EXPECT_NE(dynamics.getTerraHelper(), nullptr);
  EXPECT_EQ(dynamics.getTerraHelper()->getLSTMModel()->stream_, stream);
}

TEST_F(BicycleSlipHybridTest, computeDynamicsCPUZeroNetworks)
{
  auto dynamics = BicycleSlipHybrid();

  auto params = dynamics.getParams();
  params.max_steer_angle = 5.0;
  params.wheel_base = 2.981;
  dynamics.setParams(params);

  BicycleSlipHybrid::state_array x = BicycleSlipHybrid::state_array::Zero();
  BicycleSlipHybrid::control_array u = BicycleSlipHybrid::control_array::Zero();
  BicycleSlipHybrid::output_array output = BicycleSlipHybrid::output_array::Zero();
  dynamics.initializeDynamics(x, u, output, 0, 0);

  // computeDynamics should not touch the roll/pitch element
  BicycleSlipHybrid::state_array state_der = BicycleSlipHybrid::state_array::Ones() * 0.153;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);   // x
  EXPECT_FLOAT_EQ(state_der(1), 0);   // y
  EXPECT_FLOAT_EQ(state_der(2), 0);   // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);   // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);   // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);   // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);   // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);   // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);  // steer angle rate

  u << 1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);   // x
  EXPECT_FLOAT_EQ(state_der(1), 0);   // y
  EXPECT_FLOAT_EQ(state_der(2), 0);   // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);   // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);   // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);   // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);   // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);   // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);  // steer angle rate

  u << -1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);     // x
  EXPECT_FLOAT_EQ(state_der(1), 0);     // y
  EXPECT_FLOAT_EQ(state_der(2), 0);     // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);     // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.33);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);     // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);     // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);     // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);     // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);     // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);    // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << 1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);     // x
  EXPECT_FLOAT_EQ(state_der(1), 0);     // y
  EXPECT_FLOAT_EQ(state_der(2), 0);     // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);     // steer angle
  EXPECT_FLOAT_EQ(state_der(4), -0.9);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);     // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);     // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);     // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);     // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);     // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);    // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << -0.9, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);      // x
  EXPECT_FLOAT_EQ(state_der(1), 0);      // y
  EXPECT_FLOAT_EQ(state_der(2), 0);      // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), -0.66);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);      // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);      // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);      // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);     // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << -0.9, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);      // x
  EXPECT_FLOAT_EQ(state_der(1), 0);      // y
  EXPECT_FLOAT_EQ(state_der(2), 0);      // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), -0.66);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);      // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);      // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);      // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);     // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << -0.01, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);      // x
  EXPECT_FLOAT_EQ(state_der(1), 0);      // y
  EXPECT_FLOAT_EQ(state_der(2), 0);      // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.066);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);      // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);      // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);      // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);     // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 1.0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);    // x
  EXPECT_FLOAT_EQ(state_der(1), 0);    // y
  EXPECT_FLOAT_EQ(state_der(2), 0);    // yaw
  EXPECT_FLOAT_EQ(state_der(3), 3.0);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);    // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);    // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);    // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);    // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);    // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);    // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);   // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -1.0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);   // x
  EXPECT_FLOAT_EQ(state_der(1), 0);   // y
  EXPECT_FLOAT_EQ(state_der(2), 0);   // yaw
  EXPECT_FLOAT_EQ(state_der(3), -3);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);   // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);   // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);   // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);   // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);  // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);    // x
  EXPECT_FLOAT_EQ(state_der(1), 0);    // y
  EXPECT_FLOAT_EQ(state_der(2), 0);    // yaw
  EXPECT_FLOAT_EQ(state_der(3), 1.5);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);    // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);    // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);    // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);    // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);    // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);    // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);   // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -0.1;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);     // x
  EXPECT_FLOAT_EQ(state_der(1), 0);     // y
  EXPECT_FLOAT_EQ(state_der(2), 0);     // yaw
  EXPECT_FLOAT_EQ(state_der(3), -0.3);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);     // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);     // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);     // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);     // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);     // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);    // steer angle rate

  x << 0, 0, 0, -2.5, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);   // x
  EXPECT_FLOAT_EQ(state_der(1), 0);   // y
  EXPECT_FLOAT_EQ(state_der(2), 0);   // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);   // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);   // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);   // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);   // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);   // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);  // steer angle rate

  x << 0, 0, 0, 2.5, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);   // x
  EXPECT_FLOAT_EQ(state_der(1), 0);   // y
  EXPECT_FLOAT_EQ(state_der(2), 0);   // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);   // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);   // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);   // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);   // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);   // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);  // steer angle rate

  x << 0, 0, 0, 2.5, 0, 1.0, 1.0, 1.0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 1);    // x
  EXPECT_FLOAT_EQ(state_der(1), 1);    // y
  EXPECT_FLOAT_EQ(state_der(2), 1.0);  // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);    // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);    // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);    // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);    // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);    // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);    // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);    // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);   // steer angle rate

  x << 0, 0, M_PI_4f32, 2.5, 0, 1.0, 1.0, 1.0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                    // x
  EXPECT_FLOAT_EQ(state_der(1), cosf(M_PI_4f32) * 2);  // y
  EXPECT_FLOAT_EQ(state_der(2), 1.0);                  // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);                    // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);                    // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);                    // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);                    // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);                    // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                    // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                    // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                   // steer angle rate

  x << 0, 0, -M_PI_4f32, 2.5, 0, 1.0, 0.0, 1.0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), cosf(M_PI_4f32));   // x
  EXPECT_FLOAT_EQ(state_der(1), -cosf(M_PI_4f32));  // y
  EXPECT_FLOAT_EQ(state_der(2), 1.0);               // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);                 // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);                 // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);                 // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);                 // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);                 // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                 // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                 // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                // steer angle rate
}

TEST_F(BicycleSlipHybridTest, computeDynamicsCPUFakeNetworks)
{
  auto dynamics = BicycleSlipHybrid();

  auto params = dynamics.getParams();
  params.max_steer_angle = 5.0;
  params.wheel_base = 2.981;
  dynamics.setParams(params);

  BicycleSlipHybrid::state_array x = BicycleSlipHybrid::state_array::Zero();
  BicycleSlipHybrid::control_array u = BicycleSlipHybrid::control_array::Zero();
  BicycleSlipHybrid::output_array output = BicycleSlipHybrid::output_array::Zero();
  dynamics.initializeDynamics(x, u, output, 0, 0);

  // force brake output
  auto brake_params = dynamics.getDelayHelper()->getOutputModel()->getParams();
  std::vector<float> brake_theta(BicycleSlipHybrid::DELAY_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  brake_theta[brake_params.stride_idcs[3]] = 1.0;
  dynamics.getDelayHelper()->getOutputModel()->updateModel({ 7, 1 }, brake_theta);

  auto steer_params = dynamics.getSteerHelper()->getOutputModel()->getParams();
  std::vector<float> steer_theta(BicycleSlipHybrid::STEER_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  steer_theta[steer_params.stride_idcs[3]] = 2.0;
  dynamics.getSteerHelper()->getOutputModel()->updateModel({ 9, 5, 1 }, steer_theta);

  auto terra_params = dynamics.getTerraHelper()->getOutputModel()->getParams();
  std::vector<float> terra_theta(BicycleSlipHybrid::TERRA_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  terra_theta[terra_params.stride_idcs[3]] = 4.0;
  terra_theta[terra_params.stride_idcs[3] + 1] = 10.0;
  terra_theta[terra_params.stride_idcs[3] + 2] = 6.0;
  dynamics.getTerraHelper()->getOutputModel()->updateModel({ 32, 20, 3 }, terra_theta);

  float delta = 0;
  // computeDynamics should not touch the roll/pitch element
  BicycleSlipHybrid::state_array state_der = BicycleSlipHybrid::state_array::Ones() * 0.153;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                                      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                                                     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30);                                                      // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  u << 1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                                      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                                                     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  u << -1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                                      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.33);                                                    // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  // change the initial brake state
  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << 1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                                      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), -0.9);                                                    // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << -0.9, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                                      // steer angle
  EXPECT_NEAR(state_der(4), -0.66, tol);                                                  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << -0.9, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                                      // steer angle
  EXPECT_NEAR(state_der(4), -0.66, tol);                                                  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << -0.01, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                                      // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.066);                                                   // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 1.0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 3.0 + 20);                                                // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                                                     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -1.0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), -3 + 20);                                                 // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                                                     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 1.5 + 20);                                                // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                                                     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -0.1;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), -0.3 + 20);                                               // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                                                     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                                      // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                                      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                                                   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                                      // steer angle rate

  x << 0, 0, 0, -2.5, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -0.1;
  delta = tanf(x(3) / -9.2);
  float c_delta = cosf(delta + 0.07);
  float s_delta = sinf(delta + 0.07);
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                 // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                 // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                 // yaw
  EXPECT_FLOAT_EQ(state_der(3), 1.2 + 20);                          // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                               // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                             // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                 // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                 // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                // steer angle rate

  x << 0, 0, 0, 2.5, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 0.5;
  delta = tanf(x(3) / -9.2);
  c_delta = cosf(delta + 0.07);
  s_delta = sinf(delta + 0.07);
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                 // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                 // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                 // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                               // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                             // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                 // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                 // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                // steer angle rate

  x << 0, 0, 0, 0, 0, 1.0, 2.0, 0, 0, 0, 0;
  u << 0, 0.5;
  delta = tanf(x(3) / -9.2);
  c_delta = cosf(delta + 0.07);
  s_delta = sinf(delta + 0.07);
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 1.0);                               // x
  EXPECT_FLOAT_EQ(state_der(1), 2.0);                               // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                 // yaw
  EXPECT_FLOAT_EQ(state_der(3), 21.5);                              // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.0);                               // brake state
  EXPECT_FLOAT_EQ(state_der(5), 40);                                // vel x
  EXPECT_FLOAT_EQ(state_der(6), 50);                                // vel y
  EXPECT_FLOAT_EQ(state_der(7), 30.0f);                             // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                 // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                 // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                // steer angle rate
}

TEST_F(BicycleSlipHybridTest, updateState)
{
  auto dynamics = BicycleSlipHybrid();

  auto params = dynamics.getParams();
  params.max_steer_angle = 5.0;
  dynamics.setParams(params);

  auto limits = dynamics.getControlRanges();
  limits[0].x = -1.0;
  dynamics.setControlRanges(limits);

  BicycleSlipHybrid::state_array s = BicycleSlipHybrid::state_array::Zero();
  BicycleSlipHybrid::state_array s_next = BicycleSlipHybrid::state_array::Zero();
  BicycleSlipHybrid::state_array s_der = BicycleSlipHybrid::state_array::Zero();

  s << 1, 2, 3, 4, 0.55, 6, 7, 8, 9, 10, 11;
  dynamics.updateState(s, s_next, s_der, 0.1);
  EXPECT_FLOAT_EQ(s_next(0), 1);     // x
  EXPECT_FLOAT_EQ(s_next(1), 2);     // y
  EXPECT_FLOAT_EQ(s_next(2), 3);     // yaw
  EXPECT_FLOAT_EQ(s_next(3), 4);     // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 0.55);  // brake state
  EXPECT_FLOAT_EQ(s_next(5), 6);     // vel x
  EXPECT_FLOAT_EQ(s_next(6), 7);     // vel y
  EXPECT_FLOAT_EQ(s_next(7), 8);     // omega z
  EXPECT_FLOAT_EQ(s_next(8), 9);     // roll
  EXPECT_FLOAT_EQ(s_next(9), 10);    // pitch
  EXPECT_FLOAT_EQ(s_next(10), 0);    // steer angle rate

  s = BicycleSlipHybrid::state_array::Ones() * 10;
  dynamics.updateState(s, s_next, s_der, 0.1);
  EXPECT_FLOAT_EQ(s_next(0), 10);                                 // x
  EXPECT_FLOAT_EQ(s_next(1), 10);                                 // y
  EXPECT_FLOAT_EQ(s_next(2), angle_utils::normalizeAngle(s(2)));  // yaw
  EXPECT_FLOAT_EQ(s_next(3), 5);                                  // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 1.0);                                // brake state
  EXPECT_FLOAT_EQ(s_next(5), 10);                                 // vel x
  EXPECT_FLOAT_EQ(s_next(6), 10);                                 // vel y
  EXPECT_FLOAT_EQ(s_next(7), 10);                                 // omega z
  EXPECT_FLOAT_EQ(s_next(8), 10);                                 // roll
  EXPECT_FLOAT_EQ(s_next(9), 10);                                 // pitch
  EXPECT_FLOAT_EQ(s_next(10), 0);                                 // steer angle rate

  s = BicycleSlipHybrid::state_array::Ones() * -10;
  dynamics.updateState(s, s_next, s_der, 0.1);
  EXPECT_FLOAT_EQ(s_next(0), -10);                                // x
  EXPECT_FLOAT_EQ(s_next(1), -10);                                // y
  EXPECT_FLOAT_EQ(s_next(2), angle_utils::normalizeAngle(s(2)));  // yaw
  EXPECT_FLOAT_EQ(s_next(3), -5);                                 // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 0.0);                                // brake state
  EXPECT_FLOAT_EQ(s_next(5), -10);                                // vel x
  EXPECT_FLOAT_EQ(s_next(6), -10);                                // vel y
  EXPECT_FLOAT_EQ(s_next(7), -10);                                // omega z
  EXPECT_FLOAT_EQ(s_next(8), -10);                                // roll
  EXPECT_FLOAT_EQ(s_next(9), -10);                                // pitch
  EXPECT_FLOAT_EQ(s_next(10), 0);                                 // steer angle rate

  s = BicycleSlipHybrid::state_array::Zero();
  s_der = BicycleSlipHybrid::state_array::Ones();
  dynamics.updateState(s, s_next, s_der, 0.1);
  EXPECT_FLOAT_EQ(s_next(0), 0.1);   // x
  EXPECT_FLOAT_EQ(s_next(1), 0.1);   // y
  EXPECT_NEAR(s_next(2), 0.1, tol);  // yaw
  EXPECT_FLOAT_EQ(s_next(3), 0.1);   // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 0.1);   // brake state
  EXPECT_FLOAT_EQ(s_next(5), 0.1);   // vel x
  EXPECT_FLOAT_EQ(s_next(6), 0.1);   // vel y
  EXPECT_FLOAT_EQ(s_next(7), 0.1);   // omega z
  EXPECT_FLOAT_EQ(s_next(8), 0.1);   // roll
  EXPECT_FLOAT_EQ(s_next(9), 0.1);   // pitch
  EXPECT_FLOAT_EQ(s_next(10), 1.0);  // steer angle rate

  s = BicycleSlipHybrid::state_array::Zero();
  s_der = BicycleSlipHybrid::state_array::Ones() * -1;
  dynamics.updateState(s, s_next, s_der, 0.1);
  EXPECT_FLOAT_EQ(s_next(0), -0.1);   // x
  EXPECT_FLOAT_EQ(s_next(1), -0.1);   // y
  EXPECT_NEAR(s_next(2), -0.1, tol);  // yaw
  EXPECT_FLOAT_EQ(s_next(3), -0.1);   // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 0.0);    // brake state
  EXPECT_FLOAT_EQ(s_next(5), -0.1);   // vel x
  EXPECT_FLOAT_EQ(s_next(6), -0.1);   // vel y
  EXPECT_FLOAT_EQ(s_next(7), -0.1);   // omega z
  EXPECT_FLOAT_EQ(s_next(8), -0.1);   // roll
  EXPECT_FLOAT_EQ(s_next(9), -0.1);   // pitch
  EXPECT_FLOAT_EQ(s_next(10), -1.0);  // steer angle rate

  s = BicycleSlipHybrid::state_array::Zero();
  s_der = BicycleSlipHybrid::state_array::Ones() * 10;
  dynamics.updateState(s, s_next, s_der, 1);
  EXPECT_FLOAT_EQ(s_next(0), 10);                                   // x
  EXPECT_FLOAT_EQ(s_next(1), 10);                                   // y
  EXPECT_NEAR(s_next(2), angle_utils::normalizeAngle(10.0f), tol);  // yaw
  EXPECT_FLOAT_EQ(s_next(3), 5);                                    // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 1);                                    // brake state
  EXPECT_FLOAT_EQ(s_next(5), 10);                                   // vel x
  EXPECT_FLOAT_EQ(s_next(6), 10);                                   // vel y
  EXPECT_FLOAT_EQ(s_next(7), 10);                                   // omega z
  EXPECT_FLOAT_EQ(s_next(8), 10);                                   // roll
  EXPECT_FLOAT_EQ(s_next(9), 10);                                   // pitch
  EXPECT_FLOAT_EQ(s_next(10), 10.0);                                // steer angle rate

  s = BicycleSlipHybrid::state_array::Zero();
  s_der = BicycleSlipHybrid::state_array::Ones() * -10;
  dynamics.updateState(s, s_next, s_der, 1.0);
  EXPECT_FLOAT_EQ(s_next(0), -10.0);                                 // x
  EXPECT_FLOAT_EQ(s_next(1), -10.0);                                 // y
  EXPECT_NEAR(s_next(2), angle_utils::normalizeAngle(-10.0f), tol);  // yaw
  EXPECT_FLOAT_EQ(s_next(3), -5.0);                                  // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 0);                                     // brake state
  EXPECT_FLOAT_EQ(s_next(5), -10.0);                                 // vel x
  EXPECT_FLOAT_EQ(s_next(6), -10.0);                                 // vel y
  EXPECT_FLOAT_EQ(s_next(7), -10.0);                                 // omega z
  EXPECT_FLOAT_EQ(s_next(8), -10);                                   // roll
  EXPECT_FLOAT_EQ(s_next(9), -10);                                   // pitch
  EXPECT_FLOAT_EQ(s_next(10), -10.0);                                // steer angle rate
}

TEST_F(BicycleSlipHybridTest, stepCPU)
{
  auto dynamics = BicycleSlipHybrid();

  auto params = dynamics.getParams();
  params.max_steer_angle = 5.0;
  params.wheel_base = 2.981;
  dynamics.setParams(params);

  auto limits = dynamics.getControlRanges();
  limits[0].x = -1.0;
  dynamics.setControlRanges(limits);

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
  helper->setExtent(0, extent);

  std::vector<float> data_vec;
  data_vec.resize(10 * 20);
  for (int i = 1; i < data_vec.size(); i++)
  {
    data_vec[i] = i * 1.0f;
  }

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, make_float3(0, 0, 0));

  helper->updateTexture(0, data_vec);
  helper->updateResolution(0, 10);
  helper->enableTexture(0);
  helper->copyToDevice(true);

  // force brake output
  auto brake_params = dynamics.getDelayHelper()->getOutputModel()->getParams();
  std::vector<float> brake_theta(BicycleSlipHybrid::DELAY_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  brake_theta[brake_params.stride_idcs[3]] = 1.0;
  dynamics.getDelayHelper()->getOutputModel()->updateModel({ 7, 1 }, brake_theta);

  auto steer_params = dynamics.getSteerHelper()->getOutputModel()->getParams();
  std::vector<float> steer_theta(BicycleSlipHybrid::STEER_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  steer_theta[steer_params.stride_idcs[3]] = 2.0;
  dynamics.getSteerHelper()->getOutputModel()->updateModel({ 9, 5, 1 }, steer_theta);

  auto terra_params = dynamics.getTerraHelper()->getOutputModel()->getParams();
  std::vector<float> terra_theta(BicycleSlipHybrid::TERRA_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  terra_theta[terra_params.stride_idcs[3]] = 4.0;
  terra_theta[terra_params.stride_idcs[3] + 1] = 10.0;
  terra_theta[terra_params.stride_idcs[3] + 2] = 6.0;
  dynamics.getTerraHelper()->getOutputModel()->updateModel({ 32, 20, 3 }, terra_theta);

  BicycleSlipHybrid::state_array s = BicycleSlipHybrid::state_array::Ones();
  BicycleSlipHybrid::control_array u = BicycleSlipHybrid::control_array::Ones();
  BicycleSlipHybrid::state_array s_next = BicycleSlipHybrid::state_array::Zero();
  BicycleSlipHybrid::state_array s_der = BicycleSlipHybrid::state_array::Ones();
  BicycleSlipHybrid::output_array output = BicycleSlipHybrid::output_array::Zero();
  s(0) = 5;
  s(1) = 5;

  dynamics.step(s, s_next, s_der, u, output, 0, 0.1);
  EXPECT_FLOAT_EQ(s_der(0), -0.30116868);  // x
  EXPECT_FLOAT_EQ(s_der(1), 1.3817732);    // y
  EXPECT_FLOAT_EQ(s_der(2), 1.0);          // yaw
  EXPECT_FLOAT_EQ(s_der(3), 22.4);         // steer angle
  EXPECT_FLOAT_EQ(s_der(4), -0.9);         // brake state
  EXPECT_FLOAT_EQ(s_der(5), 40);           // vel x
  EXPECT_FLOAT_EQ(s_der(6), 50);           // vel y
  EXPECT_FLOAT_EQ(s_der(7), 30);           // omega z
  EXPECT_FLOAT_EQ(s_der(8), 0);            // roll
  EXPECT_FLOAT_EQ(s_der(9), 0);            // pitch
  EXPECT_FLOAT_EQ(s_der(10), 0);           // steer angle rate

  EXPECT_FLOAT_EQ(s_next(0), 4.96988314);   // x
  EXPECT_FLOAT_EQ(s_next(1), 5.1381773);    // y
  EXPECT_FLOAT_EQ(s_next(2), 1.1f);         // yaw
  EXPECT_FLOAT_EQ(s_next(3), 3.24);         // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 0.91);         // brake state
  EXPECT_FLOAT_EQ(s_next(5), 4 + 1);        // vel x
  EXPECT_FLOAT_EQ(s_next(6), 5 + 1);        // vel y
  EXPECT_FLOAT_EQ(s_next(7), 3 + 1);        // omega z
  EXPECT_FLOAT_EQ(s_next(8), -0.039676614);  // roll
  EXPECT_FLOAT_EQ(s_next(9), -0.26290134);   // pitch
  EXPECT_FLOAT_EQ(s_next(10), 22.4);        // steer angle rate

  EXPECT_FLOAT_EQ(output(0), 5);            // x vel
  EXPECT_FLOAT_EQ(output(1), 6);            // y vel
  EXPECT_FLOAT_EQ(output(2), 0);            // z vel
  EXPECT_FLOAT_EQ(output(3), 4.96988314);   // x pos
  EXPECT_FLOAT_EQ(output(4), 5.1381773);    // y pos
  EXPECT_FLOAT_EQ(output(5), 0.083221436);  // z pos
  EXPECT_FLOAT_EQ(output(6), 1.1000001);    // yaw
  EXPECT_FLOAT_EQ(output(7), -0.039676614);  // roll
  EXPECT_FLOAT_EQ(output(8), -0.26290134);   // pitch
  EXPECT_FLOAT_EQ(output(9), 3.24);         // steer angle
  EXPECT_FLOAT_EQ(output(10), 22.4);        // steer angle rate
  // EXPECT_FLOAT_EQ(output(11), 5.6652329);   // fl wheel x
  // EXPECT_FLOAT_EQ(output(12), 8.1291666);   // fl wheel y
  // EXPECT_FLOAT_EQ(output(13), 6.9788725);   // fr wheel x
  // EXPECT_FLOAT_EQ(output(14), 7.460566);    // fr wheel y
  // EXPECT_FLOAT_EQ(output(15), 4.31306332);  // bl wheel x
  // EXPECT_FLOAT_EQ(output(16), 5.4724776);   // bl wheel y
  // EXPECT_FLOAT_EQ(output(17), 5.626703);    // br wheel x
  // EXPECT_FLOAT_EQ(output(18), 4.803877);    // br wheel y
  EXPECT_FLOAT_EQ(output(19), 10000);      // wheel f fl
  EXPECT_FLOAT_EQ(output(20), 10000);      // wheel f fr
  EXPECT_FLOAT_EQ(output(21), 10000);      // wheel f bl
  EXPECT_FLOAT_EQ(output(22), 10000);      // wheel f br
  EXPECT_FLOAT_EQ(output(23), 40);         // accel x
  EXPECT_FLOAT_EQ(output(24), 50);         // accel y
  EXPECT_FLOAT_EQ(output(25), 4);          // omega z
}

TEST_F(BicycleSlipHybridTest, TestPythonComparison)
{
  CudaCheckError();
  using DYN = BicycleSlipHybrid;
  BicycleSlipHybrid dynamics = BicycleSlipHybrid(mppi::tests::bicycle_slip_hybrid_test);

  auto limits = dynamics.getControlRanges();
  limits[0].x = -1.0;
  dynamics.setControlRanges(limits);

  auto params = dynamics.getParams();
  params.max_steer_angle = 5.0;
  params.wheel_base = 2.981;
  dynamics.setParams(params);

  cnpy::npz_t input_outputs = cnpy::npz_load(mppi::tests::bicycle_slip_hybrid_test);
  int num_points = input_outputs.at("num_points").data<int>()[0];
  // num_points = 1;
  float dt = input_outputs.at("dt").data<double>()[0];
  double T_temp = input_outputs.at("T").data<double>()[0];
  int T = std::round(T_temp / dt);
  double tau = input_outputs.at("tau").data<double>()[0];
  int init_T = std::round(tau / dt) + 1;
  int input_dim = input_outputs.at("input_dim").data<int>()[0];
  int output_dim = input_outputs.at("output_dim").data<int>()[0];
  EXPECT_EQ(num_points, 100);
  EXPECT_FLOAT_EQ(dt, 0.02);
  EXPECT_EQ(T, 250);
  EXPECT_EQ(init_T, 51);
  EXPECT_EQ(input_dim, 12);
  EXPECT_EQ(output_dim, 5);

  double* inputs = input_outputs.at("input").data<double>();
  double* outputs = input_outputs.at("output").data<double>();
  double* init_inputs = input_outputs.at("init_input").data<double>();
  double* delay_init_hidden = input_outputs.at("init/delay/hidden").data<double>();
  double* delay_init_cell = input_outputs.at("init/delay/cell").data<double>();
  double* steer_init_hidden = input_outputs.at("init/steer/hidden").data<double>();
  double* steer_init_cell = input_outputs.at("init/steer/cell").data<double>();
  double* terra_init_hidden = input_outputs.at("init/bicycle/terra/hidden").data<double>();
  double* terra_init_cell = input_outputs.at("init/bicycle/terra/cell").data<double>();

  // steering model params
  EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_angle, 5);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 3.5885217);
  EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 2.550503);

  // delay model params
  EXPECT_FLOAT_EQ(dynamics.getParams().brake_delay_constant, 1.6213325);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_brake_rate_neg, 0.69062573);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_brake_rate_pos, 0.16684119);

  // rest params
  EXPECT_FLOAT_EQ(dynamics.getParams().gravity, -9.81);
  EXPECT_FLOAT_EQ(dynamics.getParams().wheel_base, 2.981);
  EXPECT_FLOAT_EQ(dynamics.getParams().steer_angle_scale, -15.966150283813477);

  EXPECT_FLOAT_EQ(dynamics.getParams().c_t[0], 2.007661819458008);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_t[1], 3.480222702026367);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_b[0], 126.236328125);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_b[1], 21.20456314086914);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_v[0], 14.369324684143066);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_v[1], 0.10648054629564285);
  EXPECT_FLOAT_EQ(dynamics.getParams().c_0, 0.0);

  EXPECT_FLOAT_EQ(dynamics.getParams().ay_angle, -17.342659);
  EXPECT_FLOAT_EQ(dynamics.getParams().ay_ax[0], 0);
  EXPECT_FLOAT_EQ(dynamics.getParams().ay_ax[1], -0.5456691980361938);
  EXPECT_FLOAT_EQ(dynamics.getParams().ay_v[0], -0.38541162);
  EXPECT_FLOAT_EQ(dynamics.getParams().ay_v[1], 1.3749259);

  EXPECT_FLOAT_EQ(dynamics.getParams().omega_v[0], 0.6870362758636475);
  EXPECT_FLOAT_EQ(dynamics.getParams().omega_v[1], 0.5812708139419556);

  std::map<std::string, Eigen::VectorXf> buffer;
  buffer["VEL_X"] = Eigen::VectorXf::Random(51);
  buffer["VEL_Y"] = Eigen::VectorXf::Random(51);
  buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
  buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
  buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);
  buffer["BRAKE_STATE"] = Eigen::VectorXf::Random(51);
  buffer["BRAKE_CMD"] = Eigen::VectorXf::Random(51);
  buffer["THROTTLE_CMD"] = Eigen::VectorXf::Random(51);
  buffer["OMEGA_Z"] = Eigen::VectorXf::Random(51);
  buffer["ROLL"] = Eigen::VectorXf::Random(51);
  buffer["PITCH"] = Eigen::VectorXf::Random(51);

  BicycleSlipHybrid::state_array state;
  BicycleSlipHybrid::state_array next_state_cpu;
  BicycleSlipHybrid::control_array control;
  BicycleSlipHybrid::output_array output;
  BicycleSlipHybrid::state_array state_der = BicycleSlipHybrid::state_array::Zero();

  for (int point = 0; point < num_points; point++)
  {
    for (int t = 0; t < init_T; t++)
    {
      buffer["VEL_X"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 0];
      buffer["VEL_Y"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 1];
      buffer["OMEGA_Z"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 2];
      buffer["THROTTLE_CMD"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 3];
      buffer["BRAKE_STATE"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 4];
      buffer["STEER_ANGLE"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 5];
      buffer["STEER_ANGLE_RATE"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 6];
      buffer["PITCH"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 7];
      buffer["ROLL"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 8];
      buffer["BRAKE_CMD"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 9];
      buffer["STEER_CMD"](t) = init_inputs[point * init_T * input_dim + t * input_dim + 10];
    }
    dynamics.updateFromBuffer(buffer);

    for (int i = 0; i < BicycleSlipKinematic::DELAY_LSTM::HIDDEN_DIM; i++)
    {
      EXPECT_NEAR(dynamics.getDelayHelper()->getLSTMModel()->getHiddenState()(i),
                  delay_init_hidden[BicycleSlipKinematic::DELAY_LSTM::HIDDEN_DIM * point + i], tol)
          << "at point " << point << " index " << i;
      EXPECT_NEAR(dynamics.getDelayHelper()->getLSTMModel()->getCellState()(i),
                  delay_init_cell[BicycleSlipKinematic::DELAY_LSTM::HIDDEN_DIM * point + i], tol)
          << "at point " << point << " index " << i;
    }
    for (int i = 0; i < BicycleSlipKinematic::STEER_LSTM::HIDDEN_DIM; i++)
    {
      EXPECT_NEAR(dynamics.getSteerHelper()->getLSTMModel()->getHiddenState()(i),
                  steer_init_hidden[BicycleSlipKinematic::STEER_LSTM::HIDDEN_DIM * point + i], tol)
          << "at point " << point << " index " << i;
      EXPECT_NEAR(dynamics.getSteerHelper()->getLSTMModel()->getCellState()(i),
                  steer_init_cell[BicycleSlipKinematic::STEER_LSTM::HIDDEN_DIM * point + i], tol)
          << "at point " << point << " index " << i;
    }
    for (int i = 0; i < BicycleSlipKinematic::TERRA_LSTM::HIDDEN_DIM; i++)
    {
      EXPECT_NEAR(dynamics.getTerraHelper()->getLSTMModel()->getHiddenState()(i),
                  terra_init_hidden[BicycleSlipKinematic::TERRA_LSTM::HIDDEN_DIM * point + i], tol)
          << "at point " << point << " index " << i;
      EXPECT_NEAR(dynamics.getTerraHelper()->getLSTMModel()->getCellState()(i),
                  terra_init_cell[BicycleSlipKinematic::TERRA_LSTM::HIDDEN_DIM * point + i], tol)
          << "at point " << point << " index " << i;
    }

    BicycleSlipHybrid::state_array state;
    for (int t = 0; t < T; t++)
    {
      state = BicycleSlipKinematic::state_array::Zero();
      state_der = BicycleSlipKinematic::state_array::Zero();
      state(3) = inputs[point * T * input_dim + t * input_dim + 5];   // STEER_ANGLE
      state(4) = inputs[point * T * input_dim + t * input_dim + 4];   // BRAKE_STATE
      state(5) = inputs[point * T * input_dim + t * input_dim + 0];   // VX
      state(6) = inputs[point * T * input_dim + t * input_dim + 1];   // VY
      state(7) = inputs[point * T * input_dim + t * input_dim + 2];   // OMEGA_Z
      state(8) = inputs[point * T * input_dim + t * input_dim + 8];   // ROLL
      state(9) = inputs[point * T * input_dim + t * input_dim + 7];   // PITCH
      state(10) = inputs[point * T * input_dim + t * input_dim + 6];  // STEER_ANGLE_RATE
      control(0) = inputs[point * T * input_dim + t * input_dim + 3] -
                   inputs[point * T * input_dim + t * input_dim + 9];   // THROTTLE/BRAKE
      control(1) = inputs[point * T * input_dim + t * input_dim + 10];  // STEER_CMD

      dynamics.step(state, next_state_cpu, state_der, control, output, 0, dt);

      EXPECT_NEAR(state_der[5], outputs[point * T * output_dim + t * output_dim + 0], tol)
          << "point " << point << " at dim ACCEL_X at time " << t;
      EXPECT_NEAR(state_der[6], outputs[point * T * output_dim + t * output_dim + 1], tol)
          << "point " << point << " at dim ACCEL_Y"
          << " at time " << t;
      EXPECT_NEAR(state_der[7], outputs[point * T * output_dim + t * output_dim + 2], tol)
          << "point " << point << " at dim OMEGA_Z"
          << " at time " << t;
      EXPECT_NEAR(state_der[4], outputs[point * T * output_dim + t * output_dim + 3], tol)
          << "point " << point << " at dim BRAKE_STATE"
          << " at time " << t;
      EXPECT_NEAR(state_der[3], outputs[point * T * output_dim + t * output_dim + 4], tol)
          << "point " << point << " at dim STEER_ANGLE"
          << " at time " << t;
      // for (int i = 0; i < 25; i++)
      // {
      //   EXPECT_NEAR(dynamics.getLSTMModel()->getHiddenState()[i], hidden[point * T * 25 + 25 * t + i], tol)
      //                 << "point " << point << " at dim " << i;
      //   EXPECT_NEAR(dynamics.getLSTMModel()->getCellState()[i], cell[point * T * 25 + 25 * t + i], tol)
      //                 << "point " << point << " at dim " << i;
      // }
    }
  }
}

TEST_F(BicycleSlipHybridTest, TestStepGPUvsCPU)
{
  const int num_rollouts = 2000;
  const float dt = 0.1f;
  CudaCheckError();
  using DYN = BicycleSlipHybrid;
  BicycleSlipHybrid dynamics = BicycleSlipHybrid(mppi::tests::bicycle_slip_hybrid_test);

  auto params = dynamics.getParams();
  params.max_steer_angle = 5.0;
  params.wheel_base = 2.981;
  dynamics.setParams(params);

  cudaExtent extent = make_cudaExtent(100, 200, 0);
  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
  helper->setExtent(0, extent);

  std::vector<float> data_vec;
  data_vec.resize(100 * 200);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = i * 0.1f;
  }

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, make_float3(0, 0, 0));

  helper->updateTexture(0, data_vec);
  helper->updateResolution(0, 10);
  helper->enableTexture(0);
  helper->copyToDevice(true);

  CudaCheckError();
  dynamics.GPUSetup();
  CudaCheckError();

  EXPECT_NE(dynamics.getSteerHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.steer_network_d_, nullptr);
  EXPECT_EQ(dynamics.steer_network_d_, dynamics.getSteerHelper()->getLSTMDevicePtr());

  EXPECT_NE(dynamics.getDelayHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.delay_network_d_, nullptr);
  EXPECT_EQ(dynamics.delay_network_d_, dynamics.getDelayHelper()->getLSTMDevicePtr());

  EXPECT_NE(dynamics.getTerraHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.terra_network_d_, nullptr);
  EXPECT_EQ(dynamics.terra_network_d_, dynamics.getTerraHelper()->getLSTMDevicePtr());

  Eigen::Matrix<float, BicycleSlipHybrid::CONTROL_DIM, num_rollouts> control_trajectory;
  control_trajectory = Eigen::Matrix<float, BicycleSlipHybrid::CONTROL_DIM, num_rollouts>::Random();
  Eigen::Matrix<float, BicycleSlipHybrid::STATE_DIM, num_rollouts> state_trajectory;
  state_trajectory = Eigen::Matrix<float, BicycleSlipHybrid::STATE_DIM, num_rollouts>::Random();

  std::vector<std::array<float, BicycleSlipHybrid::STATE_DIM>> s(num_rollouts);
  std::vector<std::array<float, BicycleSlipHybrid::STATE_DIM>> s_next(num_rollouts);
  std::vector<std::array<float, BicycleSlipHybrid::STATE_DIM>> s_der(num_rollouts);
  // steering, throttle
  std::vector<std::array<float, BicycleSlipHybrid::CONTROL_DIM>> u(num_rollouts);

  BicycleSlipHybrid::state_array state;
  BicycleSlipHybrid::state_array next_state_cpu;
  BicycleSlipHybrid::control_array control;
  BicycleSlipHybrid::output_array output;
  BicycleSlipHybrid::state_array state_der_cpu = BicycleSlipHybrid::state_array::Zero();

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for (int y_dim = 1; y_dim <= 16; y_dim++)
  {
    DYN::buffer_trajectory buffer;
    buffer["VEL_X"] = Eigen::VectorXf::Random(51);
    buffer["VEL_Y"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_ANGLE_RATE"] = Eigen::VectorXf::Random(51);
    buffer["STEER_CMD"] = Eigen::VectorXf::Random(51);
    buffer["BRAKE_STATE"] = Eigen::VectorXf::Random(51);
    buffer["BRAKE_CMD"] = Eigen::VectorXf::Random(51);
    buffer["THROTTLE_CMD"] = Eigen::VectorXf::Random(51);
    buffer["OMEGA_Z"] = Eigen::VectorXf::Random(51);
    buffer["ROLL"] = Eigen::VectorXf::Random(51);
    buffer["PITCH"] = Eigen::VectorXf::Random(51);

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
    launchStepTestKernel<BicycleSlipHybrid, 16>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
    for (int point = 0; point < num_rollouts; point++)
    {
      dynamics.initializeDynamics(state, control, output, 0, 0);
      state = state_trajectory.col(point);
      control = control_trajectory.col(point);
      state_der_cpu = BicycleSlipHybrid::state_array::Zero();

      dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
      for (int dim = 0; dim < BicycleSlipHybrid::STATE_DIM - 1; dim++)
      {
        EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        EXPECT_NEAR(next_state_cpu(dim), s_next[point][dim], 1e-4)
            << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
        EXPECT_TRUE(isfinite(s_next[point][dim]));
      }
    }
  }
  dynamics.freeCudaMem();
}

// TEST_F(BicycleSlipHybridTest, TestStepGPUvsCPUReverse)
// {
//   using DYN = BicycleSlipHybrid;
//
//   const int num_rollouts = 2000;
//   const float dt = 0.1f;
//   CudaCheckError();
//   BicycleSlipHybrid dynamics = BicycleSlipHybrid(mppi::tests::steering_lstm);
//   auto params = dynamics.getParams();
//   params.gear_sign = -1;
//   dynamics.setParams(params);
//   EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 3.9760568141937256);
//   EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 2.1222121715545654);
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
//   Eigen::Matrix<float, BicycleSlipHybrid::CONTROL_DIM, num_rollouts> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, BicycleSlipHybrid::CONTROL_DIM, num_rollouts>::Random();
//   Eigen::Matrix<float, BicycleSlipHybrid::STATE_DIM, num_rollouts> state_trajectory;
//   state_trajectory = Eigen::Matrix<float, BicycleSlipHybrid::STATE_DIM, num_rollouts>::Random();
//
//   std::vector<std::array<float, BicycleSlipHybrid::STATE_DIM>> s(num_rollouts);
//   std::vector<std::array<float, BicycleSlipHybrid::STATE_DIM>> s_next(num_rollouts);
//   std::vector<std::array<float, BicycleSlipHybrid::STATE_DIM>> s_der(num_rollouts);
//   // steering, throttle
//   std::vector<std::array<float, BicycleSlipHybrid::CONTROL_DIM>> u(num_rollouts);
//
//   BicycleSlipHybrid::state_array state;
//   BicycleSlipHybrid::state_array next_state_cpu;
//   BicycleSlipHybrid::control_array control;
//   BicycleSlipHybrid::output_array output;
//   BicycleSlipHybrid::state_array state_der_cpu = BicycleSlipHybrid::state_array::Zero();
//
//   // Run dynamics on dynamicsU
//   // Run dynamics on GPU
//   for (int y_dim = 1; y_dim <= 16; y_dim++)
//   {
//     DYN::buffer_trajectory buffer;
//     buffer["VEL_X"] = Eigen::VectorXf::Random(51);
//     buffer["VEL_Y"] = Eigen::VectorXf::Random(51);
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
//     launchStepTestKernel<BicycleSlipHybrid>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
//     for (int point = 0; point < num_rollouts; point++)
//     {
//       dynamics.initializeDynamics(state, control, output, 0, 0);
//       state = state_trajectory.col(point);
//       control = control_trajectory.col(point);
//       state_der_cpu = BicycleSlipHybrid::state_array::Zero();
//
//       dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//       // for (int dim = 0; dim < BicycleSlipHybrid::STATE_DIM; dim++)
//       for (int dim = 0; dim < BicycleSlipHybrid::STATE_DIM; dim++)
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
// /*
// class LinearDummy : public BicycleSlipHybrid {
// public:
//   bool computeGrad(const Eigen::Ref<const state_array> & state,
//                    const Eigen::Ref<const control_array>& control,
//                    Eigen::Ref<dfdx> A,
//                    Eigen::Ref<dfdu> B) {
//     return false;
//   };
// };
//
// TEST_F(BicycleSlipHybridTest, TestComputeGradComputation) {
//   Eigen::Matrix<float, BicycleSlipHybrid::STATE_DIM, BicycleSlipHybrid::STATE_DIM +
// BicycleSlipHybrid::CONTROL_DIM> numeric_jac; Eigen::Matrix<float,
// BicycleSlipHybrid::STATE_DIM, BicycleSlipHybrid::STATE_DIM +
// BicycleSlipHybrid::CONTROL_DIM> analytic_jac; BicycleSlipHybrid::state_array state;
// state
// << 1, 2, 3, 4; BicycleSlipHybrid::control_array control; control << 5;
//
//   auto analytic_grad_model = BicycleSlipHybrid();
//
//   BicycleSlipHybrid::dfdx A_analytic = BicycleSlipHybrid::dfdx::Zero();
//   BicycleSlipHybrid::dfdu B_analytic = BicycleSlipHybrid::dfdu::Zero();
//
//   analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);
//
//   auto numerical_grad_model = LinearDummy();
//
//   std::shared_ptr<ModelWrapperDDP<LinearDummy>> ddp_model =
// std::make_shared<ModelWrapperDDP<LinearDummy>>(&numerical_grad_model);
//
//   analytic_jac.leftCols<BicycleSlipHybrid::STATE_DIM>() = A_analytic;
//   analytic_jac.rightCols<BicycleSlipHybrid::CONTROL_DIM>() = B_analytic;
//   numeric_jac = ddp_model->df(state, control);
//
//   ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n" << numeric_jac << "\nAnalytic
//   Jacobian\n"
// << analytic_jac;
// }
//
// */
