#include <gtest/gtest.h>
#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>

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
  // Noise enters the system during the "true" state propagation. In this case the noise is nominal

  // Initialize the double integrator dynamics and cost
  DoubleIntegratorDynamics model;
  DoubleIntegratorCircleCost cost;
  float dt = 0.01; // Timestep of dynamics propagation
  int max_iter = 10; // Maximum running iterations of optimization
  float gamma = 0.5; // Learning rate parameter
  const int num_timesteps = 100;  // Optimization time horizon

  int total_time_horizon = 1000; // Problem time horizon

  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 0;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          512, 64, 8>(&model, &cost, dt, max_iter, gamma, num_timesteps, control_var);


  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t) {
    // Print the system state
    model.printState(x.data());

    // Compute the control
    vanilla_controller.computeControl(x);

    // Propagate the state forward
    model.computeDynamics(x, vanilla_controller.getControlSeq().col(0), xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);
  }
}

TEST(TubeMPPITest, VanillaMPPILargeVariance) {
// Noise enters the system during the "true" state propagation. In this case the noise is nominal

  // Initialize the double integrator dynamics and cost
  DoubleIntegratorDynamics model(10);
  DoubleIntegratorCircleCost cost;
  float dt = 0.01; // Timestep of dynamics propagation
  int max_iter = 10; // Maximum running iterations of optimization
  float gamma = 0.5; // Learning rate parameter
  const int num_timesteps = 100;  // Optimization time horizon

  int total_time_horizon = 1000; // Problem time horizon

  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 0;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          512, 64, 8>(&model, &cost, dt, max_iter, gamma, num_timesteps, control_var);


  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t) {
    // Print the system state
    model.printState(x.data());

    // Compute the control
    vanilla_controller.computeControl(x);

    // Propagate the state forward
    model.computeDynamics(x, vanilla_controller.getControlSeq().col(0), xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);
  }
}

TEST(TubeMPPITest, TubeMPPILargeVariance) {
  // Noise enters the system during the "true" state propagation. In this case the noise is nominal

  // Initialize the double integrator dynamics

  // Initialize the cost function

  // Initialize the tube MPPI controller

  // Start the while loop

  // Compute the control

  // Get the feedback gains associated with the nominal state and control trajectory

  // Apply the feedback given the current state

  // Propagate the state forward

  // Add the "true" noise of the system
}
