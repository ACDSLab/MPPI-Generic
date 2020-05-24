//
// Created by mgandhi on 5/23/20.
//
#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/core/rmppi_kernel_test.cuh>
#include <mppi/utils/test_helper.h>

class RMPPIKernels : public ::testing::Test {
public:
  using dynamics = DoubleIntegratorDynamics;
  using cost_function = DoubleIntegratorCircleCost;

  void SetUp() override {
    model = new dynamics(10);  // Initialize the double integrator dynamics
    cost = new cost_function;  // Initialize the cost function
  }

  void TearDown() override {
    delete model;
    delete cost;
  }

  dynamics* model;
  cost_function* cost;
};

TEST_F(RMPPIKernels, InitEvalRollout) {
  // Given the initial states, we need to roll out the number of samples.
  // 1.)  Generate the noise used to evaluate each sample.
  //
  Eigen::Matrix<float, 4, 9> x0_candidates;
  x0_candidates << -4 , -3, -2, -1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4, 4, 4,
  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // For each candidate, we want to estimate the free energy using a set number of samples.
  const int num_samples = 64;

  // We are going to propagate a trajectory for a given number of timesteps
  const int num_timesteps = 100;

  // We need to generate a nominal trajectory for the control
  auto nominal_control = Eigen::MatrixXf::Random(dynamics::CONTROL_DIM, num_timesteps);

  // Let us make temporary variables to hold the states and state derivatives and controls
  dynamics::state_array x_current, x_dot_current;
  dynamics::control_array u_current;

  Eigen::Matrix<float, 1, num_samples*9> cost_vector;

  float cost_current = 0.0;
  for (int i = 0; i < 9; ++i) { // Iterate through each candidate
  for (int j = 0; j < num_samples; ++j) {
  x_current = x0_candidates.col(i);  // The initial state of the rollout
  for (int k = 0; k < num_timesteps; ++k) {
  // compute the cost
  cost_current += cost->computeStateCost(x_current);
  // get the control plus a disturbance
  u_current = nominal_control.col(k) + Eigen::MatrixXf::Random(dynamics::CONTROL_DIM, 1);
  // compute the next state_dot
  model->computeDynamics(x_current, u_current, x_dot_current);
  // update the state to the next
  model->updateState(x_current, x_dot_current, 0.01);
  }
  // compute the terminal cost -> this is the free energy estimate, save it!
  cost_current += cost->terminalCost(x_current);
  cost_vector.col(i*num_samples + j) << cost_current;
  }
  }

  // Run the GPU test kernel of the init eval kernel and get the output data
  rmppi_kernels::launchInitEvalKernel<dynamics, cost_function, 64, 8>();
  // Compare with the CPU version
  FAIL();
}