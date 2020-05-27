#include <gtest/gtest.h>
#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>
#include <cnpy.h>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <mppi/ddp/ddp_tracking_costs.h>
#include <mppi/ddp/ddp.h>

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

const int total_time_horizon = 2500;

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
                                      512, 64, 8>(&model, &cost, dt, max_iter, gamma, control_var);

  auto controller = TubeMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
                                        512, 64, 8>(&model, &cost, dt, max_iter, gamma, Q, Q, R, control_var);

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
  float dt = 0.02; // Timestep of dynamics propagation
  int max_iter = 3; // Maximum running iterations of optimization
  float gamma = 0.25; // Learning rate parameter
  const int num_timesteps = 50;  // Optimization time horizon


  std::vector<float> nominal_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);

  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          1024, 64, 8>(&model, &cost, dt, max_iter, gamma, control_var);

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

    if (cost.computeStateCost(x) > 1000) {
      fail_count++;
    }

    if (tubeFailure(x.data())) {
      FAIL();
    }

    // Compute the control
    vanilla_controller.computeControl(x);

    // Save the nominal trajectory
    auto nominal_trajectory = vanilla_controller.getStateSeq();

    for (int i = 0; i < num_timesteps; i++) {
      for (int j = 0; j < DoubleIntegratorDynamics::STATE_DIM; j++) {
        nominal_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
        i*DoubleIntegratorDynamics::STATE_DIM + j] = nominal_trajectory(j, i);
      }
    }

    // Propagate the state forward
    model.computeDynamics(x, vanilla_controller.getControlSeq().col(0), xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    vanilla_controller.slideControlSequence(1);
  }
//  std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
//save it to file
  cnpy::npy_save("vanilla_nominal.npy",nominal_trajectory_save.data(),
          {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
}

TEST(TubeMPPITest, VanillaMPPILargeVariance) {
// Noise enters the system during the "true" state propagation. In this case the noise is nominal

  // Initialize the double integrator dynamics and cost
  DoubleIntegratorDynamics model(100);
  DoubleIntegratorCircleCost cost;
  float dt = 0.02; // Timestep of dynamics propagation
  int max_iter = 3; // Maximum running iterations of optimization
  float gamma = 0.25; // Learning rate parameter
  const int num_timesteps = 50;  // Optimization time horizon

  std::vector<float> nominal_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);


  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          1024, 64, 8>(&model, &cost, dt, max_iter, gamma, control_var);

  //bool success = false;
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

    if (tubeFailure(x.data()))  {
      //success = true;
      fail_count++;
    }

    if (fail_count > 50) {
      break;
    }
    // Compute the control
    vanilla_controller.computeControl(x);

    // Save the nominal trajectory
    auto nominal_trajectory = vanilla_controller.getStateSeq();

    for (int i = 0; i < num_timesteps; i++) {
      for (int j = 0; j < DoubleIntegratorDynamics::STATE_DIM; j++) {
        nominal_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                                i*DoubleIntegratorDynamics::STATE_DIM + j] = nominal_trajectory(j, i);
      }
    }

    // Propagate the state forward
    model.computeDynamics(x, vanilla_controller.getControlSeq().col(0), xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    vanilla_controller.slideControlSequence(1);
//    if (success) {
//      break;
//    }
  }

  cnpy::npy_save("vanilla_large.npy",nominal_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
//  std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
}

TEST(TubeMPPITest, VanillaMPPILargeVarianceTracking) {
// Noise enters the system during the "true" state propagation. In this case the noise is nominal

  // Initialize the double integrator dynamics and cost
  DoubleIntegratorDynamics model(100);
  DoubleIntegratorCircleCost cost;
  float dt = 0.02; // Timestep of dynamics propagation
  int max_iter = 3; // Maximum running iterations of optimization
  float gamma = 0.25; // Learning rate parameter
  const int num_timesteps = 50;  // Optimization time horizon

  std::vector<float> nominal_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);


  // Set the initial state
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;

  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          1024, 64, 8>(&model, &cost, dt, max_iter, gamma, control_var);

  // DDP cost parameters
  Eigen::MatrixXf Q;
  Eigen::MatrixXf Qf;
  Eigen::MatrixXf R;

  Q = 500*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);
  Q(2,2) = 100;
  Q(3,3) = 100;
  R = 1*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::CONTROL_DIM,DoubleIntegratorDynamics::CONTROL_DIM);

  Qf = Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);

  vanilla_controller.initDDP(Q, Qf, R);

  //bool success = false;
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

    if (tubeFailure(x.data()))  {
      //success = true;
      fail_count++;
    }

    if (fail_count > 50) {
      break;
    }
    // Compute the control
    vanilla_controller.computeControl(x);

    // Save the nominal trajectory
    auto nominal_trajectory = vanilla_controller.getStateSeq();
    auto nominal_control = vanilla_controller.getControlSeq();

    for (int i = 0; i < num_timesteps; i++) {
      for (int j = 0; j < DoubleIntegratorDynamics::STATE_DIM; j++) {
        nominal_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                                i*DoubleIntegratorDynamics::STATE_DIM + j] = nominal_trajectory(j, i);
      }
    }

    // Compute the feedback gains
    vanilla_controller.computeFeedbackGains(x);

    // Get the open loop control
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);
    //    std::cout << current_control << std::endl;

    // Apply the feedback given the current state
    current_control += vanilla_controller.getFeedbackGains()[0]*(x - nominal_trajectory.col(0));

    // Propagate the state forward
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    vanilla_controller.slideControlSequence(1);
  }

  cnpy::npy_save("vanilla_large_track.npy",nominal_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
//  std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
}

TEST(TubeMPPITest, TubeMPPILargeVariance) {
  // Noise enters the system during the "true" state propagation. In this case the noise is nominal
  DoubleIntegratorDynamics model(100);  // Initialize the double integrator dynamics
  DoubleIntegratorCircleCost cost;  // Initialize the cost function
  float dt = 0.02; // Timestep of dynamics propagation
  int max_iter = 3; // Maximum running iterations of optimization
  float gamma = 0.25; // Learning rate parameter
  const int num_timesteps = 50;  // Optimization time horizon

  std::vector<float> actual_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);
  std::vector<float> nominal_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);
  std::vector<float> ancillary_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);


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

  /**
   * Q =
   * [500, 0, 0, 0
   *  0, 500, 0, 0
   *  0, 0, 100, 0
   *  0, 0, 0, 100]
   */
  Q = 500*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);
  Q(2,2) = 100;
  Q(3,3) = 100;
  /**
   * R = I
   */
  R = 1*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::CONTROL_DIM,DoubleIntegratorDynamics::CONTROL_DIM);

  /**
   * Qf = I
   */
  Qf = Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);

  // Initialize the tube MPPI controller
  auto controller = TubeMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, num_timesteps,
          1024, 64, 1>(&model, &cost, dt, max_iter, gamma, Q, Qf, R, control_var);

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

    if (cost.computeStateCost(x) > 1000) {
      fail_count++;
    }

    if (tubeFailure(x.data())) {
      cnpy::npy_save("tube_large_actual.npy", actual_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
      cnpy::npy_save("tube_ancillary.npy", ancillary_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
      cnpy::npy_save("tube_large_nominal.npy",nominal_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
      FAIL() << "Visualize the trajectories by running scripts/double_integrator/plot_DI_test_trajectories; "
               "the argument to this python file is the build directory of MPPI-Generic";
    }

    // Compute the control
    controller.computeControl(x);

    // Save the trajectory from the nominal state
    auto nominal_trajectory = controller.getStateSeq();
    auto actual_trajectory = controller.getActualStateSeq();

    // Get the feedback gains associated with the nominal state and control trajectory
    controller.computeFeedbackGains(x);

    // Save the ancillary trajectory
    auto ancillary_trajectory = controller.getAncillaryStateSeq();

    for (int i = 0; i < num_timesteps; i++) {
      for (int j = 0; j < DoubleIntegratorDynamics::STATE_DIM; j++) {
        actual_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                                i*DoubleIntegratorDynamics::STATE_DIM + j] = actual_trajectory(j, i);
        ancillary_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                                i*DoubleIntegratorDynamics::STATE_DIM + j] = ancillary_trajectory(j, i);
        nominal_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                               i*DoubleIntegratorDynamics::STATE_DIM + j] = nominal_trajectory(j, i);
      }
    }

    // Get the open loop control
    DoubleIntegratorDynamics::control_array current_control = controller.getActualControlSeq().col(0);
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

  cnpy::npy_save("tube_large_actual.npy",actual_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
  cnpy::npy_save("tube_ancillary.npy",ancillary_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
  cnpy::npy_save("tube_large_nominal.npy",nominal_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
}
