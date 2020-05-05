#include <gtest/gtest.h>
#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>

TEST(RMPPITest, CPURolloutKernel) {
  DoubleIntegratorDynamics model;
  DoubleIntegratorCircleCost cost;
  float dt = 0.01;
  int max_iter = 10;
  float gamma = 0.5;
  const int num_timesteps = 100;
}