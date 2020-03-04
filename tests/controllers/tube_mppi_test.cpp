#include <gtest/gtest.h>
#include <instantiations/double_integrator_mppi/double_integrator_mppi.cuh>

TEST(TubeMPPITest, Construction) {
  // Define the model and cost
  DoubleIntegratorDynamics model;
  DoubleIntegratorCircleCost cost;
  float dt = 0.01;
  int max_iter = 10;
  float gamma = 0.5;
  const int num_timesteps = 100;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // DDP cost parameters
  Eigen::MatrixXf Q;
  Eigen::MatrixXf R;

  Q = 100*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);
  R = Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::CONTROL_DIM,DoubleIntegratorDynamics::CONTROL_DIM);

  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
                                      512, 64, 8>(&model, &cost, dt, max_iter, gamma, num_timesteps, control_var);

  auto controller = TubeMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
                                        512, 64, 8>();

//  auto controller = TubeMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
//                                      512, 64, 8>(&model, &cost, dt, max_iter,
//                                                   gamma, num_timesteps, Q, Q, R, control_var);

  // This controller needs the ancillary controller running separately for base plant reasons.



}

TEST(TubeMPPITest, VanillaMPPINominalVariance) {
  // Noise enters the system during the "true" state propagation.
}

TEST(TubeMPPITest, VanillaMPPILargeVariance) {

}
