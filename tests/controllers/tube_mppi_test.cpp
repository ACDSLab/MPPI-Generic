#include <gtest/gtest.h>
#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>

bool tubeFailure(float *s) {
  float inner_path_radius2 = 1.675*1.675;
  float outer_path_radius2 = 2.325*2.325;
  float radial_position = s[0]*s[0] + s[1]*s[1];
  if ((radial_position < inner_path_radius2) || (radial_position > outer_path_radius2)) {
    return true;
  } else {
    return false;
  }
}

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
                                        512, 64, 8>(&model, &cost, dt, max_iter, gamma, num_timesteps, Q, Q, R, control_var);

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
  int max_iter = 1; // Maximum running iterations of optimization
  float gamma = 0.25; // Learning rate parameter
  const int num_timesteps = 100;  // Optimization time horizon

  int total_time_horizon = 1000; // Problem time horizon

  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          512, 64, 8>(&model, &cost, dt, max_iter, gamma, num_timesteps, control_var);

  int fail_count = 0;
  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t) {
    // Print the system state
//    if (t % 100 == 0) {
////      float current_cost = cost.getStateCost(x.data());
//      printf("Current Time: %f    ", t * dt);
////      printf("Current State Cost: %f    ", current_cost);
//      model.printState(x.data());
//    }

    if (cost.getStateCost(x.data()) > 1000) {
      fail_count++;
    }

    if (tubeFailure(x.data())) {
      FAIL();
    }

    // Compute the control
    vanilla_controller.computeControl(x);

    // Propagate the state forward
    model.computeDynamics(x, vanilla_controller.getControlSeq().col(0), xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    vanilla_controller.slideControlSequence(1);
  }
//  std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
}

TEST(TubeMPPITest, VanillaMPPILargeVariance) {
// Noise enters the system during the "true" state propagation. In this case the noise is nominal

  // Initialize the double integrator dynamics and cost
  DoubleIntegratorDynamics model(100);
  DoubleIntegratorCircleCost cost;
  float dt = 0.01; // Timestep of dynamics propagation
  int max_iter = 1; // Maximum running iterations of optimization
  float gamma = 0.25; // Learning rate parameter
  const int num_timesteps = 100;  // Optimization time horizon

  int total_time_horizon = 1000; // Problem time horizon

  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          512, 64, 8>(&model, &cost, dt, max_iter, gamma, num_timesteps, control_var);

  bool success = false;
  int fail_count = 0;

  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t) {
    // Print the system state
//    if (t % 100 == 0) {
//      float current_cost = cost.getStateCost(x.data());
//      printf("Current Time: %f    ", t * dt);
//      printf("Current State Cost: %f    ", current_cost);
//      model.printState(x.data());
//    }

    if (cost.getStateCost(x.data()) > 1000) {
      fail_count++;
      success = true;
    }

    if (tubeFailure(x.data())) {
      success = true;
    }

    // Compute the control
    vanilla_controller.computeControl(x);

    // Propagate the state forward
    model.computeDynamics(x, vanilla_controller.getControlSeq().col(0), xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    vanilla_controller.slideControlSequence(1);
    if (success) {
      break;
    }
  }
//  std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
  if (not success) {
    FAIL();
  }
}

TEST(TubeMPPITest, TubeMPPILargeVariance) {
  // Noise enters the system during the "true" state propagation. In this case the noise is nominal
  DoubleIntegratorDynamics model(100);  // Initialize the double integrator dynamics
  DoubleIntegratorCircleCost cost;  // Initialize the cost function
  float dt = 0.01; // Timestep of dynamics propagation
  int max_iter = 1; // Maximum running iterations of optimization
  float gamma = 0.25; // Learning rate parameter
  const int num_timesteps = 100;  // Optimization time horizon

  int total_time_horizon = 1000; // Problem time horizon


  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // DDP cost parameters
  Eigen::MatrixXf Q;
  Eigen::MatrixXf Qf;
  Eigen::MatrixXf R;

  Q = 500*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);
  Q(2,2) = 100;
  Q(3,3) = 100;
  R = 1*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::CONTROL_DIM,DoubleIntegratorDynamics::CONTROL_DIM);

  Qf = Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);

  // Initialize the tube MPPI controller
  auto controller = TubeMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          512, 64, 8>(&model, &cost, dt, max_iter, gamma, num_timesteps, Q, Qf, R, control_var);

  int fail_count = 0;

  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t) {
    // Print the system state
//    if (t % 100 == 0) {
//      float current_cost = cost.getStateCost(x.data());
//      printf("Current Time: %f    ", t * dt);
//      printf("Current State Cost: %f    ", current_cost);
//      model.printState(x.data());
//    }

    if (cost.getStateCost(x.data()) > 1000) {
      fail_count++;
    }

    if (tubeFailure(x.data())) {
      FAIL();
    }

    // Compute the control
    controller.computeControl(x);

    // Get the feedback gains associated with the nominal state and control trajectory
    controller.computeFeedbackGains(x);

    // Get the open loop control
    DoubleIntegratorDynamics::control_array current_control = controller.getControlSeq().col(0);
//    std::cout << current_control << std::endl;


    // Apply the feedback given the current state
    current_control += controller.getFeedbackGains()[0]*(x - controller.getStateSeq().col(0));

//    std::cout << "Current State: " << x.transpose() << std::endl;
//    std::cout << "Nominal State: " << controller.getStateSeq().col(0).transpose()  << std::endl;


    // Propagate the state forward
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    controller.updateNominalState(controller.getControlSeq().col(0));

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    controller.slideControlSequence(1);
  }
}
