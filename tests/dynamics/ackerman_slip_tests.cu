#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/ackerman_slip/ackerman_slip.cuh>
#include <mppi/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <autorally_test_network.h>
#include <cuda_runtime.h>

class AckermanSlipTest : public ::testing::Test
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

const double tol = 1e-6;

TEST_F(AckermanSlipTest, Template)
{
  auto dynamics = AckermanSlip();
  EXPECT_EQ(11, AckermanSlip::STATE_DIM);
  EXPECT_EQ(2, AckermanSlip::CONTROL_DIM);
  EXPECT_TRUE(dynamics.checkRequiresBuffer());
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);

  EXPECT_NE(dynamics.getSteerHelper(), nullptr);
  EXPECT_NE(dynamics.getDelayHelper(), nullptr);
  EXPECT_NE(dynamics.getEngineHelper(), nullptr);
  EXPECT_NE(dynamics.getTerraHelper(), nullptr);
}

TEST_F(AckermanSlipTest, BindStream)
{
  auto dynamics = AckermanSlip(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);

  EXPECT_NE(dynamics.getSteerHelper(), nullptr);
  EXPECT_EQ(dynamics.getSteerHelper()->getLSTMModel()->stream_, stream);
  EXPECT_NE(dynamics.getDelayHelper(), nullptr);
  EXPECT_EQ(dynamics.getDelayHelper()->getLSTMModel()->stream_, stream);
  EXPECT_NE(dynamics.getEngineHelper(), nullptr);
  EXPECT_EQ(dynamics.getEngineHelper()->getLSTMModel()->stream_, stream);
  EXPECT_NE(dynamics.getTerraHelper(), nullptr);
  EXPECT_EQ(dynamics.getTerraHelper()->getLSTMModel()->stream_, stream);
}

TEST_F(AckermanSlipTest, computeDynamicsCPUZeroNetworks)
{
  auto dynamics = AckermanSlip();

  AckermanSlip::state_array x = AckermanSlip::state_array::Zero();
  AckermanSlip::control_array u = AckermanSlip::control_array::Zero();
  AckermanSlip::output_array output = AckermanSlip::output_array::Zero();
  dynamics.initializeDynamics(x, u, output, 0, 0);

  // computeDynamics should not touch the roll/pitch element
  AckermanSlip::state_array state_der = AckermanSlip::state_array::Ones() * 0.153;
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
  EXPECT_FLOAT_EQ(state_der(0), 1);   // x
  EXPECT_FLOAT_EQ(state_der(1), 1);   // y
  EXPECT_FLOAT_EQ(state_der(2), 1);   // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);   // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);   // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);   // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);   // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);   // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);   // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);  // steer angle rate

  x << 0, 0, M_PI_4f32, 2.5, 0, 1.0, 1.0, 1.0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                    // x
  EXPECT_FLOAT_EQ(state_der(1), cosf(M_PI_4f32) * 2);  // y
  EXPECT_FLOAT_EQ(state_der(2), 1);                    // yaw
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
  EXPECT_FLOAT_EQ(state_der(2), 1);                 // yaw
  EXPECT_FLOAT_EQ(state_der(3), 0);                 // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0);                 // brake state
  EXPECT_FLOAT_EQ(state_der(5), 0);                 // vel x
  EXPECT_FLOAT_EQ(state_der(6), 0);                 // vel y
  EXPECT_FLOAT_EQ(state_der(7), 0);                 // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                 // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                 // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                // steer angle rate
}

TEST_F(AckermanSlipTest, computeDynamicsCPUFakeNetworks)
{
  auto dynamics = AckermanSlip();

  AckermanSlip::state_array x = AckermanSlip::state_array::Zero();
  AckermanSlip::control_array u = AckermanSlip::control_array::Zero();
  AckermanSlip::output_array output = AckermanSlip::output_array::Zero();
  dynamics.initializeDynamics(x, u, output, 0, 0);

  // force brake output
  auto brake_params = dynamics.getDelayHelper()->getOutputModel()->getParams();
  std::vector<float> brake_theta(AckermanSlip::DELAY_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  brake_theta[brake_params.stride_idcs[3]] = 1.0;
  dynamics.getDelayHelper()->getOutputModel()->updateModel({ 8, 30, 1 }, brake_theta);

  auto steer_params = dynamics.getSteerHelper()->getOutputModel()->getParams();
  std::vector<float> steer_theta(AckermanSlip::STEER_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  steer_theta[steer_params.stride_idcs[3]] = 2.0;
  dynamics.getSteerHelper()->getOutputModel()->updateModel({ 11, 20, 1 }, steer_theta);

  auto engine_params = dynamics.getEngineHelper()->getOutputModel()->getParams();
  std::vector<float> engine_theta(AckermanSlip::ENGINE_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  engine_theta[engine_params.stride_idcs[3]] = 3.0;
  dynamics.getEngineHelper()->getOutputModel()->updateModel({ 8, 20, 1 }, engine_theta);

  auto terra_params = dynamics.getTerraHelper()->getOutputModel()->getParams();
  std::vector<float> terra_theta(AckermanSlip::TERRA_LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  terra_theta[terra_params.stride_idcs[3]] = 4.0;
  terra_theta[terra_params.stride_idcs[3] + 1] = 5.0;
  terra_theta[terra_params.stride_idcs[3] + 2] = 6.0;
  dynamics.getTerraHelper()->getOutputModel()->updateModel({ 18, 20, 3 }, terra_theta);

  float delta = 0;
  // computeDynamics should not touch the roll/pitch element
  AckermanSlip::state_array state_der = AckermanSlip::state_array::Ones() * 0.153;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);        // x
  EXPECT_FLOAT_EQ(state_der(1), 0);        // y
  EXPECT_FLOAT_EQ(state_der(2), 0);        // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);       // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);      // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);  // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);       // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);        // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);        // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);       // steer angle rate

  u << 1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);        // x
  EXPECT_FLOAT_EQ(state_der(1), 0);        // y
  EXPECT_FLOAT_EQ(state_der(2), 0);        // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);       // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);      // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);  // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);       // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);        // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);        // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);       // steer angle rate

  u << -1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);        // x
  EXPECT_FLOAT_EQ(state_der(1), 0);        // y
  EXPECT_FLOAT_EQ(state_der(2), 0);        // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);       // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.33);     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);  // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);       // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);        // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);        // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);       // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << 1, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);         // x
  EXPECT_FLOAT_EQ(state_der(1), 0);         // y
  EXPECT_FLOAT_EQ(state_der(2), 0);         // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);        // steer angle
  EXPECT_FLOAT_EQ(state_der(4), -0.9 + 1);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);       // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);        // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);         // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);         // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);        // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << -0.9, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);           // x
  EXPECT_FLOAT_EQ(state_der(1), 0);           // y
  EXPECT_FLOAT_EQ(state_der(2), 0);           // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);          // steer angle
  EXPECT_NEAR(state_der(4), -0.66 + 1, tol);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);     // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);         // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);          // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);           // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);           // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);          // steer angle rate

  x << 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0;
  u << -0.9, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);           // x
  EXPECT_FLOAT_EQ(state_der(1), 0);           // y
  EXPECT_FLOAT_EQ(state_der(2), 0);           // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);          // steer angle
  EXPECT_NEAR(state_der(4), -0.66 + 1, tol);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);     // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);         // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);          // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);           // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);           // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);          // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << -0.01, 0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);          // x
  EXPECT_FLOAT_EQ(state_der(1), 0);          // y
  EXPECT_FLOAT_EQ(state_der(2), 0);          // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);         // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 0.066 + 1);  // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);    // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);        // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);         // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);          // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);          // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);         // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 1.0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);         // x
  EXPECT_FLOAT_EQ(state_der(1), 0);         // y
  EXPECT_FLOAT_EQ(state_der(2), 0);         // yaw
  EXPECT_FLOAT_EQ(state_der(3), 3.0 + 20);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);       // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);       // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);        // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);         // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);         // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);        // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -1.0;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);        // x
  EXPECT_FLOAT_EQ(state_der(1), 0);        // y
  EXPECT_FLOAT_EQ(state_der(2), 0);        // yaw
  EXPECT_FLOAT_EQ(state_der(3), -3 + 20);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);      // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);  // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);      // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);       // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);        // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);        // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);       // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 0.5;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);         // x
  EXPECT_FLOAT_EQ(state_der(1), 0);         // y
  EXPECT_FLOAT_EQ(state_der(2), 0);         // yaw
  EXPECT_FLOAT_EQ(state_der(3), 1.5 + 20);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);       // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);   // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);       // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);        // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);         // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);         // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);        // steer angle rate

  x << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -0.1;
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);          // x
  EXPECT_FLOAT_EQ(state_der(1), 0);          // y
  EXPECT_FLOAT_EQ(state_der(2), 0);          // yaw
  EXPECT_FLOAT_EQ(state_der(3), -0.3 + 20);  // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);        // brake state
  EXPECT_FLOAT_EQ(state_der(5), 60 - 40);    // vel x
  EXPECT_FLOAT_EQ(state_der(6), -50);        // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);         // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);          // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);          // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);         // steer angle rate

  x << 0, 0, 0, -2.5, 0, 0, 0, 0, 0, 0, 0;
  u << 0, -0.1;
  delta = tanf(x(3) / -9.2);
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                       // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                       // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                       // yaw
  EXPECT_FLOAT_EQ(state_der(3), 1.2 + 20);                                // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);                                     // brake state
  EXPECT_FLOAT_EQ(state_der(5), 30 * cosf(tanf(-2.5 / -9.2)) + 30 - 40);  // vel x
  EXPECT_FLOAT_EQ(state_der(6), 30 * sinf(tanf(-2.5 / -9.2)) - 50);       // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);                                      // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                       // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                       // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                      // steer angle rate

  x << 0, 0, 0, 2.5, 0, 0, 0, 0, 0, 0, 0;
  u << 0, 0.5;
  delta = tanf(x(3) / -9.2);
  dynamics.computeDynamics(x, u, state_der);
  EXPECT_FLOAT_EQ(state_der(0), 0);                                      // x
  EXPECT_FLOAT_EQ(state_der(1), 0);                                      // y
  EXPECT_FLOAT_EQ(state_der(2), 0);                                      // yaw
  EXPECT_FLOAT_EQ(state_der(3), 20);                                     // steer angle
  EXPECT_FLOAT_EQ(state_der(4), 1.0);                                    // brake state
  EXPECT_FLOAT_EQ(state_der(5), 30 * cosf(tanf(2.5 / -9.2)) + 30 - 40);  // vel x
  EXPECT_FLOAT_EQ(state_der(6), 30 * sinf(tanf(2.5 / -9.2)) - 50);       // vel y
  EXPECT_FLOAT_EQ(state_der(7), 60);                                     // omega z
  EXPECT_FLOAT_EQ(state_der(8), 0);                                      // roll
  EXPECT_FLOAT_EQ(state_der(9), 0);                                      // pitch
  EXPECT_FLOAT_EQ(state_der(10), 0);                                     // steer angle rate
}

TEST_F(AckermanSlipTest, updateState)
{
  auto dynamics = AckermanSlip();

  AckermanSlip::state_array s = AckermanSlip::state_array::Zero();
  AckermanSlip::state_array s_next = AckermanSlip::state_array::Zero();
  AckermanSlip::state_array s_der = AckermanSlip::state_array::Zero();

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

  s = AckermanSlip::state_array::Ones() * 10;
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

  s = AckermanSlip::state_array::Ones() * -10;
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

  s = AckermanSlip::state_array::Zero();
  s_der = AckermanSlip::state_array::Ones();
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

  s = AckermanSlip::state_array::Zero();
  s_der = AckermanSlip::state_array::Ones() * -1;
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

  s = AckermanSlip::state_array::Zero();
  s_der = AckermanSlip::state_array::Ones() * 10;
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

  s = AckermanSlip::state_array::Zero();
  s_der = AckermanSlip::state_array::Ones() * -10;
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

TEST_F(AckermanSlipTest, stepCPU)
{
  auto dynamics = AckermanSlip();

  AckermanSlip::state_array s = AckermanSlip::state_array::Ones();
  AckermanSlip::control_array u = AckermanSlip::control_array::Ones();
  AckermanSlip::state_array s_next = AckermanSlip::state_array::Zero();
  AckermanSlip::state_array s_der = AckermanSlip::state_array::Ones();
  AckermanSlip::output_array output = AckermanSlip::output_array::Zero();

  dynamics.step(s, s_next, s_der, u, output, 0, 0.1);
  EXPECT_FLOAT_EQ(s_der(0), -0.30116868);  // x
  EXPECT_FLOAT_EQ(s_der(1), 1.3817732);    // y
  EXPECT_FLOAT_EQ(s_der(2), 1);            // yaw
  EXPECT_FLOAT_EQ(s_der(3), 2.4);          // steer angle
  EXPECT_FLOAT_EQ(s_der(4), -.9);          // brake state
  EXPECT_FLOAT_EQ(s_der(5), 0);            // vel x
  EXPECT_FLOAT_EQ(s_der(6), 0);            // vel y
  EXPECT_FLOAT_EQ(s_der(7), 0);            // omega z
  EXPECT_FLOAT_EQ(s_der(8), 0);            // roll
  EXPECT_FLOAT_EQ(s_der(9), 0);            // pitch
  EXPECT_FLOAT_EQ(s_der(10), 0);           // steer angle rate

  EXPECT_FLOAT_EQ(s_next(0), 0.96988314);  // x
  EXPECT_FLOAT_EQ(s_next(1), 1.1381773);   // y
  EXPECT_FLOAT_EQ(s_next(2), 1.1);         // yaw
  EXPECT_FLOAT_EQ(s_next(3), 1.24);        // steer angle
  EXPECT_FLOAT_EQ(s_next(4), 0.91);        // brake state
  EXPECT_FLOAT_EQ(s_next(5), 1.0);         // vel x
  EXPECT_FLOAT_EQ(s_next(6), 1.0);         // vel y
  EXPECT_FLOAT_EQ(s_next(7), 1.0);         // omega z
  EXPECT_FLOAT_EQ(s_next(8), 0);           // roll
  EXPECT_FLOAT_EQ(s_next(9), 0);           // pitch
  EXPECT_FLOAT_EQ(s_next(10), 2.4);        // steer angle rate

  EXPECT_FLOAT_EQ(output(0), 1);            // x vel
  EXPECT_FLOAT_EQ(output(1), 1);            // y vel
  EXPECT_FLOAT_EQ(output(2), 0);            // z vel
  EXPECT_FLOAT_EQ(output(3), 0.96988314);   // x pos
  EXPECT_FLOAT_EQ(output(4), 1.1381773);    // y pos
  EXPECT_FLOAT_EQ(output(5), 0);            // z pos
  EXPECT_FLOAT_EQ(output(6), 1.1);          // yaw
  EXPECT_FLOAT_EQ(output(7), 0);            // roll
  EXPECT_FLOAT_EQ(output(8), 0);            // pitch
  EXPECT_FLOAT_EQ(output(9), 1.24);         // steer angle
  EXPECT_FLOAT_EQ(output(10), 2.4);         // steer angle rate
  EXPECT_FLOAT_EQ(output(11), 1.6652329);   // fl wheel x
  EXPECT_FLOAT_EQ(output(12), 4.1291666);   // fl wheel y
  EXPECT_FLOAT_EQ(output(13), 2.9788725);   // fr wheel x
  EXPECT_FLOAT_EQ(output(14), 3.460566);    // fr wheel y
  EXPECT_FLOAT_EQ(output(15), 0.31306332);  // bl wheel x
  EXPECT_FLOAT_EQ(output(16), 1.4724776);   // bl wheel y
  EXPECT_FLOAT_EQ(output(17), 1.626703);    // br wheel x
  EXPECT_FLOAT_EQ(output(18), 0.803877);    // br wheel y
  EXPECT_FLOAT_EQ(output(19), 10000);       // wheel f fl
  EXPECT_FLOAT_EQ(output(20), 10000);       // wheel f fr
  EXPECT_FLOAT_EQ(output(21), 10000);       // wheel f bl
  EXPECT_FLOAT_EQ(output(22), 10000);       // wheel f br
  EXPECT_FLOAT_EQ(output(23), 0);           // accel x
  EXPECT_FLOAT_EQ(output(24), 0);           // accel y
}

TEST_F(AckermanSlipTest, TestStepGPUvsCPU)
{
  const int num_rollouts = 2000;
  const float dt = 0.1f;
  CudaCheckError();
  using DYN = AckermanSlip;
  AckermanSlip dynamics = AckermanSlip(mppi::tests::steering_lstm, mppi::tests::ackerman_lstm);
  // steering model params
  EXPECT_FLOAT_EQ(dynamics.getParams().max_steer_rate, 3.9760568141937256);
  EXPECT_FLOAT_EQ(dynamics.getParams().steering_constant, 2.1222121715545654);

  // delay model params
  EXPECT_FLOAT_EQ(dynamics.getParams().brake_delay_constant, 6.7566104);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_brake_rate_neg, 1.0109744);
  EXPECT_FLOAT_EQ(dynamics.getParams().max_brake_rate_pos, 0.22115348);

  // rest params
  EXPECT_FLOAT_EQ(dynamics.getParams().gravity, -9.6544762);
  EXPECT_FLOAT_EQ(dynamics.getParams().wheel_angle_scale, -9.2476206);

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
  helper->setExtent(0, extent);

  std::vector<float> data_vec;
  data_vec.resize(10 * 20);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = i * 1.0f;
  }

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, make_float3(1, 2, 3));

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

  EXPECT_NE(dynamics.getEngineHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.engine_network_d_, nullptr);
  EXPECT_EQ(dynamics.engine_network_d_, dynamics.getEngineHelper()->getLSTMDevicePtr());

  EXPECT_NE(dynamics.getTerraHelper()->getLSTMDevicePtr(), nullptr);
  EXPECT_NE(dynamics.terra_network_d_, nullptr);
  EXPECT_EQ(dynamics.terra_network_d_, dynamics.getTerraHelper()->getLSTMDevicePtr());

  Eigen::Matrix<float, AckermanSlip::CONTROL_DIM, num_rollouts> control_trajectory;
  control_trajectory = Eigen::Matrix<float, AckermanSlip::CONTROL_DIM, num_rollouts>::Random();
  Eigen::Matrix<float, AckermanSlip::STATE_DIM, num_rollouts> state_trajectory;
  state_trajectory = Eigen::Matrix<float, AckermanSlip::STATE_DIM, num_rollouts>::Random();

  std::vector<std::array<float, AckermanSlip::STATE_DIM>> s(num_rollouts);
  std::vector<std::array<float, AckermanSlip::STATE_DIM>> s_next(num_rollouts);
  std::vector<std::array<float, AckermanSlip::STATE_DIM>> s_der(num_rollouts);
  // steering, throttle
  std::vector<std::array<float, AckermanSlip::CONTROL_DIM>> u(num_rollouts);

  AckermanSlip::state_array state;
  AckermanSlip::state_array next_state_cpu;
  AckermanSlip::control_array control;
  AckermanSlip::output_array output;
  AckermanSlip::state_array state_der_cpu = AckermanSlip::state_array::Zero();

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
    launchStepTestKernel<AckermanSlip>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
    for (int point = 0; point < num_rollouts; point++)
    {
      dynamics.initializeDynamics(state, control, output, 0, 0);
      state = state_trajectory.col(point);
      control = control_trajectory.col(point);
      state_der_cpu = AckermanSlip::state_array::Zero();

      dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
      // for (int dim = 0; dim < AckermanSlip::STATE_DIM; dim++)
      for (int dim = 0; dim < AckermanSlip::STATE_DIM; dim++)
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
//
// TEST_F(AckermanSlipTest, TestStepGPUvsCPUReverse)
// {
//   using DYN = AckermanSlip;
//
//   const int num_rollouts = 2000;
//   const float dt = 0.1f;
//   CudaCheckError();
//   AckermanSlip dynamics = AckermanSlip(mppi::tests::steering_lstm);
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
//   Eigen::Matrix<float, AckermanSlip::CONTROL_DIM, num_rollouts> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, AckermanSlip::CONTROL_DIM, num_rollouts>::Random();
//   Eigen::Matrix<float, AckermanSlip::STATE_DIM, num_rollouts> state_trajectory;
//   state_trajectory = Eigen::Matrix<float, AckermanSlip::STATE_DIM, num_rollouts>::Random();
//
//   std::vector<std::array<float, AckermanSlip::STATE_DIM>> s(num_rollouts);
//   std::vector<std::array<float, AckermanSlip::STATE_DIM>> s_next(num_rollouts);
//   std::vector<std::array<float, AckermanSlip::STATE_DIM>> s_der(num_rollouts);
//   // steering, throttle
//   std::vector<std::array<float, AckermanSlip::CONTROL_DIM>> u(num_rollouts);
//
//   AckermanSlip::state_array state;
//   AckermanSlip::state_array next_state_cpu;
//   AckermanSlip::control_array control;
//   AckermanSlip::output_array output;
//   AckermanSlip::state_array state_der_cpu = AckermanSlip::state_array::Zero();
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
//     launchStepTestKernel<AckermanSlip>(dynamics, s, u, s_der, s_next, 0, dt, y_dim);
//     for (int point = 0; point < num_rollouts; point++)
//     {
//       dynamics.initializeDynamics(state, control, output, 0, 0);
//       state = state_trajectory.col(point);
//       control = control_trajectory.col(point);
//       state_der_cpu = AckermanSlip::state_array::Zero();
//
//       dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//       // for (int dim = 0; dim < AckermanSlip::STATE_DIM; dim++)
//       for (int dim = 0; dim < AckermanSlip::STATE_DIM; dim++)
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
// TEST_F(AckermanSlipTest, compareToElevationWithoutSteering)
// {
//   // by default the network will output zeros and not effect any states
//   using DYN = AckermanSlip;
//
//   const int num_rollouts = 3000;
//   const float dt = 0.1f;
//   CudaCheckError();
//   AckermanSlip dynamics = AckermanSlip();
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
//   Eigen::Matrix<float, AckermanSlip::CONTROL_DIM, num_rollouts> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, AckermanSlip::CONTROL_DIM, num_rollouts>::Random();
//   Eigen::Matrix<float, AckermanSlip::STATE_DIM, num_rollouts> state_trajectory;
//   state_trajectory = Eigen::Matrix<float, AckermanSlip::STATE_DIM, num_rollouts>::Random();
//
//   AckermanSlip::state_array state;
//   AckermanSlip::state_array next_state_cpu;
//   AckermanSlip::control_array control;
//   AckermanSlip::output_array output;
//   AckermanSlip::state_array state_der_cpu = AckermanSlip::state_array::Zero();
//
//   AckermanSlip::state_array state2;
//   AckermanSlip::state_array next_state_cpu2;
//   AckermanSlip::control_array control2;
//   AckermanSlip::output_array output2;
//   AckermanSlip::state_array state_der_cpu2 = AckermanSlip::state_array::Zero();
//
//   DYN::buffer_trajectory buffer;
//   buffer["VEL_X"] = Eigen::VectorXf::Random(51);
//   buffer["VEL_Y"] = Eigen::VectorXf::Random(51);
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
//     state_der_cpu = AckermanSlip::state_array::Zero();
//
//     dynamics2.initializeDynamics(state2, control2, output2, 0, 0);
//     state2 = state_trajectory.col(point);
//     control2 = control_trajectory.col(point);
//     state_der_cpu2 = AckermanSlip::state_array::Zero();
//
//     dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//     dynamics2.step(state2, next_state_cpu2, state_der_cpu2, control2, output2, 0, dt);
//
//     for (int dim = 0; dim < AckermanSlip::STATE_DIM; dim++)
//     {
//       EXPECT_NEAR(state_der_cpu(dim), state_der_cpu2(dim), 1e-4) << "at index " << point << " dim " << dim;
//       EXPECT_NEAR(next_state_cpu(dim), next_state_cpu2(dim), 1e-4) << "at index " << point << " dim " << dim;
//     }
//     for (int dim = 0; dim < AckermanSlip::OUTPUT_DIM; dim++)
//     {
//       EXPECT_NEAR(output(dim), output2(dim), 1e-4) << "at index " << point << " dim " << dim;
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
//     state_der_cpu = AckermanSlip::state_array::Zero();
//
//     dynamics2.initializeDynamics(state2, control2, output2, 0, 0);
//     state2 = state_trajectory.col(point);
//     control2 = control_trajectory.col(point);
//     state_der_cpu2 = AckermanSlip::state_array::Zero();
//
//     dynamics.step(state, next_state_cpu, state_der_cpu, control, output, 0, dt);
//     dynamics2.step(state2, next_state_cpu2, state_der_cpu2, control2, output2, 0, dt);
//
//     for (int dim = 0; dim < AckermanSlip::STATE_DIM; dim++)
//     {
//       EXPECT_NEAR(state_der_cpu(dim), state_der_cpu2(dim), 1e-4) << "at index " << point << " dim " << dim;
//       EXPECT_NEAR(next_state_cpu(dim), next_state_cpu2(dim), 1e-4) << "at index " << point << " dim " << dim;
//     }
//     for (int dim = 0; dim < AckermanSlip::OUTPUT_DIM; dim++)
//     {
//       EXPECT_NEAR(output(dim), output2(dim), 1e-4) << "at index " << point << " dim " << dim;
//     }
//   }
//   dynamics.freeCudaMem();
// }
//
// /*
// class LinearDummy : public AckermanSlip {
// public:
//   bool computeGrad(const Eigen::Ref<const state_array> & state,
//                    const Eigen::Ref<const control_array>& control,
//                    Eigen::Ref<dfdx> A,
//                    Eigen::Ref<dfdu> B) {
//     return false;
//   };
// };
//
// TEST_F(AckermanSlipTest, TestComputeGradComputation) {
//   Eigen::Matrix<float, AckermanSlip::STATE_DIM, AckermanSlip::STATE_DIM +
// AckermanSlip::CONTROL_DIM> numeric_jac; Eigen::Matrix<float,
// AckermanSlip::STATE_DIM, AckermanSlip::STATE_DIM +
// AckermanSlip::CONTROL_DIM> analytic_jac; AckermanSlip::state_array state; state
// << 1, 2, 3, 4; AckermanSlip::control_array control; control << 5;
//
//   auto analytic_grad_model = AckermanSlip();
//
//   AckermanSlip::dfdx A_analytic = AckermanSlip::dfdx::Zero();
//   AckermanSlip::dfdu B_analytic = AckermanSlip::dfdu::Zero();
//
//   analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);
//
//   auto numerical_grad_model = LinearDummy();
//
//   std::shared_ptr<ModelWrapperDDP<LinearDummy>> ddp_model =
// std::make_shared<ModelWrapperDDP<LinearDummy>>(&numerical_grad_model);
//
//   analytic_jac.leftCols<AckermanSlip::STATE_DIM>() = A_analytic;
//   analytic_jac.rightCols<AckermanSlip::CONTROL_DIM>() = B_analytic;
//   numeric_jac = ddp_model->df(state, control);
//
//   ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n" << numeric_jac << "\nAnalytic
//   Jacobian\n"
// << analytic_jac;
// }
//
// */
