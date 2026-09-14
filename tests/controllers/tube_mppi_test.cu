#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/controllers/Tube-MPPI/tube_mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/core/mppi_common.cuh>
#include <cnpy.h>

bool tubeFailure(float* s)
{
  float inner_path_radius2 = 1.675 * 1.675;
  float outer_path_radius2 = 2.325 * 2.325;
  float radial_position = s[0] * s[0] + s[1] * s[1];
  if ((radial_position < inner_path_radius2) || (radial_position > outer_path_radius2))
  {
    return true;
  }
  else
  {
    return false;
  }
}

class DoubleIntegratorTubeMPPI : public ::testing::Test
{
public:
  const int num_timesteps = 100;
  const int num_rollouts = 512;
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  using FB_CONTROLLER = DDPFeedback<DYN>;
  using SAMPLING = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
  using TUBE_CONTROLLER = TubeMPPIController<DYN, COST, FB_CONTROLLER, SAMPLING>;

  void SetUp() override
  {
    model = std::make_shared<DYN>();
    cost = std::make_shared<COST>();
    sampler = std::make_shared<SAMPLING>();
    fb_controller = std::make_shared<FB_CONTROLLER>(model.get(), dt);
    auto fb_params = fb_controller->getParams();

    // DDP cost parameters
    Eigen::MatrixXf Q;
    Eigen::MatrixXf R;
    fb_params.Q = 100 * FB_CONTROLLER::square_state_matrix::Identity();
    fb_params.Q_f = fb_params.Q;
    fb_params.R = FB_CONTROLLER::square_control_matrix::Identity();
    fb_controller->setParams(fb_params);
  }

  void TearDown() override
  {
  }

  float dt = 0.01;
  int max_iter = 10;
  float lambda = 0.5;
  float alpha = 0.0;

  std::shared_ptr<DYN> model;
  std::shared_ptr<COST> cost;
  std::shared_ptr<SAMPLING> sampler;
  std::shared_ptr<FB_CONTROLLER> fb_controller;
};

TEST_F(DoubleIntegratorTubeMPPI, Construction)
{
  auto controller = std::make_shared<TUBE_CONTROLLER>(model.get(), cost.get(), fb_controller.get(), sampler.get(), dt,
                                                      max_iter, lambda, alpha, num_timesteps, num_rollouts);
  // This controller needs the ancillary controller running separately for base plant reasons.
  auto nominal_control_trajectory = controller->getControlSeq();
  auto real_control_trajectory = controller->getActualControlSeq();
  auto nominal_state_trajectory = controller->getTargetStateSeq();
  auto real_state_trajectory = controller->getActualStateSeq();

  EXPECT_EQ(controller->getFeedbackEnabled(), true);
  EXPECT_EQ(controller->getDt(), dt);
  EXPECT_EQ(controller->getNumIters(), max_iter);
  EXPECT_EQ(controller->getLambda(), lambda);
  EXPECT_EQ(controller->getAlpha(), alpha);
  EXPECT_EQ(controller->getNumTimesteps(), num_timesteps);
  EXPECT_EQ(nominal_control_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(real_control_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(nominal_state_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(real_state_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(controller->getNumRollouts(), num_rollouts);
}

TEST_F(DoubleIntegratorTubeMPPI, ConstructionUsingParams)
{
  TUBE_CONTROLLER::TEMPLATED_PARAMS controller_params;
  controller_params.dt_ = dt;
  controller_params.num_iters_ = max_iter;
  controller_params.lambda_ = lambda;
  controller_params.alpha_ = alpha;
  controller_params.num_timesteps_ = num_timesteps;
  controller_params.num_rollouts_ = num_rollouts;
  controller_params.init_control_traj_ = TUBE_CONTROLLER::control_trajectory::Random(DYN::CONTROL_DIM, 1);
  controller_params.nominal_threshold_ = 31.0f;

  auto controller =
      std::make_shared<TUBE_CONTROLLER>(model.get(), cost.get(), fb_controller.get(), sampler.get(), controller_params);
  // This controller needs the ancillary controller running separately for base plant reasons.
  auto nominal_control_trajectory = controller->getControlSeq();
  auto real_control_trajectory = controller->getActualControlSeq();
  auto nominal_state_trajectory = controller->getTargetStateSeq();
  auto real_state_trajectory = controller->getActualStateSeq();
  auto new_controller_params = controller->getParams();

  EXPECT_EQ(controller->getFeedbackEnabled(), true);
  EXPECT_EQ(controller->getDt(), dt);
  EXPECT_EQ(controller->getNumIters(), max_iter);
  EXPECT_EQ(controller->getLambda(), lambda);
  EXPECT_EQ(controller->getAlpha(), alpha);
  EXPECT_EQ(controller->getNumTimesteps(), num_timesteps);
  EXPECT_EQ(nominal_control_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(real_control_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(nominal_state_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(real_state_trajectory.cols(), controller->getNumTimesteps());
  EXPECT_EQ(controller->getNumRollouts(), num_rollouts);
  EXPECT_FLOAT_EQ(controller->getNominalThreshold(), 31.0f);

  for (int t = 0; t < controller->getNumTimesteps(); t++)
  {
    EXPECT_FLOAT_EQ(fabsf((nominal_control_trajectory.col(t) - new_controller_params.init_control_traj_.col(t)).sum()),
                    0.0f);
    EXPECT_FLOAT_EQ(fabsf((real_control_trajectory.col(t) - new_controller_params.init_control_traj_.col(t)).sum()),
                    0.0f);
  }
}

TEST_F(DoubleIntegratorTubeMPPI, UpdateNominalThreshold)
{
  auto controller = std::make_shared<TUBE_CONTROLLER>(model.get(), cost.get(), fb_controller.get(), sampler.get(), dt,
                                                      max_iter, lambda, alpha, num_timesteps, num_rollouts);
  float orig_nominal_threshold = controller->getNominalThreshold();
  float new_nominal_threshold = orig_nominal_threshold + 3.14f;
  controller->setNominalThreshold(new_nominal_threshold);
  EXPECT_NE(controller->getNominalThreshold(), orig_nominal_threshold);
  EXPECT_FLOAT_EQ(controller->getNominalThreshold(), new_nominal_threshold);
}

TEST_F(DoubleIntegratorTubeMPPI, UpdateLengthOfNominalTrajectories)
{
  TUBE_CONTROLLER::TEMPLATED_PARAMS controller_params;
  controller_params.dt_ = dt;
  controller_params.num_iters_ = max_iter;
  controller_params.lambda_ = lambda;
  controller_params.alpha_ = alpha;
  controller_params.num_timesteps_ = num_timesteps;
  controller_params.num_rollouts_ = num_rollouts;
  controller_params.init_control_traj_ = TUBE_CONTROLLER::control_trajectory::Random(DYN::CONTROL_DIM, 1);
  controller_params.slide_control_scale_ = TUBE_CONTROLLER::control_array::Ones();

  auto controller =
      std::make_shared<TUBE_CONTROLLER>(model.get(), cost.get(), fb_controller.get(), sampler.get(), controller_params);
  int new_num_timesteps = controller->getNumTimesteps() + 5;
  controller->setNumTimesteps(new_num_timesteps);
  // This controller needs the ancillary controller running separately for base plant reasons.
  auto nominal_control_trajectory = controller->getControlSeq();
  auto real_control_trajectory = controller->getActualControlSeq();
  auto nominal_state_trajectory = controller->getTargetStateSeq();
  auto real_state_trajectory = controller->getActualStateSeq();
  auto new_controller_params = controller->getParams();

  EXPECT_EQ(controller->getNumTimesteps(), new_num_timesteps);
  EXPECT_EQ(nominal_control_trajectory.cols(), new_num_timesteps);
  EXPECT_EQ(real_control_trajectory.cols(), new_num_timesteps);
  EXPECT_EQ(nominal_state_trajectory.cols(), new_num_timesteps);
  EXPECT_EQ(real_state_trajectory.cols(), new_num_timesteps);

  for (int t = 0; t < controller->getNumTimesteps(); t++)
  {
    EXPECT_FLOAT_EQ(fabsf((nominal_control_trajectory.col(t) - new_controller_params.init_control_traj_.col(t)).sum()),
                    0.0f);
    EXPECT_FLOAT_EQ(fabsf((real_control_trajectory.col(t) - new_controller_params.init_control_traj_.col(t)).sum()),
                    0.0f);
  }
}

class DoubleIntegratorTracking : public ::testing::Test
{
public:
  static const int num_timesteps = 50;
  static const int num_rollouts = 1024;
  const unsigned int total_time_horizon = 500;
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  using FB_CONTROLLER = DDPFeedback<DYN>;
  using SAMPLING = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
  using VANILLA_CONTROLLER = VanillaMPPIController<DYN, COST, FB_CONTROLLER>;
  using TUBE_CONTROLLER = TubeMPPIController<DYN, COST, FB_CONTROLLER>;

  DYN model;
  COST cost;
  float dt = 0.02;   // Timestep of dynamics propagation
  int max_iter = 3;  // Maximum running iterations of optimization
  float lambda = 4;  // Learning rate parameter
  float alpha = 0.0;
  FB_CONTROLLER fb_controller = FB_CONTROLLER(&model, dt);
  SAMPLING sampler;

  DYN::state_array x, xdot;

  void SetUp() override
  {
    auto params = cost.getParams();
    params.velocity_desired = 2;
    cost.setParams(params);

    x << 2, 0, 0, 1;
    // control std dev
    SAMPLING::SAMPLING_PARAMS_T sampler_params;
    for (int i = 0; i < DYN::CONTROL_DIM; i++)
    {
      sampler_params.std_dev[i] = 1;
      sampler_params.control_cost_coeff[i] = 1.0;
    }
    sampler = SAMPLING(sampler_params);
  }
};

TEST_F(DoubleIntegratorTracking, VanillaMPPINominalVariance)
{
  std::vector<float> nominal_trajectory_save(num_timesteps * total_time_horizon * DYN::STATE_DIM);

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VANILLA_CONTROLLER(&model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha,
                                               num_timesteps, num_rollouts);
  auto controller_params = vanilla_controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 4, 1);
  controller_params.cost_rollout_dim_ = dim3(50, 1, 1);
  // controller_params.seed_ = 42;
  vanilla_controller.setParams(controller_params);

  int fail_count = 0;
  int crash_status[1] = { 0 };
  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t)
  {
    if (cost.computeStateCost(x, t, crash_status) > 1000)
    {
      fail_count++;
      crash_status[0] = 0;
    }

    if (tubeFailure(x.data()))
    {
      FAIL();
    }

    // Compute the control
    vanilla_controller.computeControl(x, 1);

    // Save the nominal trajectory
    auto nominal_trajectory = vanilla_controller.getTargetStateSeq();

    for (int i = 0; i < num_timesteps; i++)
    {
      for (int j = 0; j < DYN::STATE_DIM; j++)
      {
        nominal_trajectory_save[t * num_timesteps * DYN::STATE_DIM + i * DYN::STATE_DIM + j] = nominal_trajectory(j, i);
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

  // save it to file
  cnpy::npy_save("vanilla_nominal.npy", nominal_trajectory_save.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  // std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
}

TEST_F(DoubleIntegratorTracking, VanillaMPPILargeVariance)
{
  std::vector<float> nominal_trajectory_save(num_timesteps * total_time_horizon * DYN::STATE_DIM);
  model.setStateVariance(100);
  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VANILLA_CONTROLLER(&model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha,
                                               num_timesteps, num_rollouts);
  auto controller_params = vanilla_controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 4, 1);
  controller_params.cost_rollout_dim_ = dim3(50, 1, 1);
  // controller_params.seed_ = 42;
  vanilla_controller.setParams(controller_params);

  // bool success = false;
  int fail_count = 0;

  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t)
  {
    if (tubeFailure(x.data()))
    {
      // success = true;
      fail_count++;
    }

    if (fail_count > 50)
    {
      break;
    }
    // Compute the control
    vanilla_controller.computeControl(x, 1);

    // Save the nominal trajectory
    auto nominal_trajectory = vanilla_controller.getTargetStateSeq();

    for (int i = 0; i < num_timesteps; i++)
    {
      for (int j = 0; j < DYN::STATE_DIM; j++)
      {
        nominal_trajectory_save[t * num_timesteps * DYN::STATE_DIM + i * DYN::STATE_DIM + j] = nominal_trajectory(j, i);
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

  cnpy::npy_save("vanilla_large.npy", nominal_trajectory_save.data(),
                 { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
  // std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
}

TEST_F(DoubleIntegratorTracking, VanillaMPPILargeVarianceTracking)
{
  std::vector<float> nominal_trajectory_save(num_timesteps * total_time_horizon * DYN::STATE_DIM);
  std::vector<float> actual_feedback_trajectory_save(num_timesteps * total_time_horizon * DYN::STATE_DIM);
  model.setStateVariance(100);

  // Initialize the vanilla MPPI controller
  auto vanilla_controller = VANILLA_CONTROLLER(&model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha,
                                               num_timesteps, num_rollouts);
  auto controller_params = vanilla_controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 4, 1);
  controller_params.cost_rollout_dim_ = dim3(50, 1, 1);
  // controller_params.seed_ = 42;
  vanilla_controller.setParams(controller_params);

  // DDP cost parameters
  auto fb_params = vanilla_controller.getFeedbackParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  vanilla_controller.setFeedbackParams(fb_params);

  // bool success = false;
  int fail_count = 0;

  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t)
  {
    if (tubeFailure(x.data()))
    {
      // success = true;
      fail_count++;
    }

    if (fail_count > 50)
    {
      break;
    }
    // Compute the control
    vanilla_controller.computeControl(x, 1);

    // Compute the feedback gains
    vanilla_controller.computeFeedback(x);

    // Save the nominal trajectory
    auto nominal_trajectory = vanilla_controller.getTargetStateSeq();
    auto nominal_control = vanilla_controller.getControlSeq();
    vanilla_controller.computeFeedbackPropagatedStateSeq();
    auto feedback_state_trajectory = vanilla_controller.getFeedbackPropagatedStateSeq();

    for (int i = 0; i < num_timesteps; i++)
    {
      for (int j = 0; j < DYN::STATE_DIM; j++)
      {
        nominal_trajectory_save[t * num_timesteps * DYN::STATE_DIM + i * DYN::STATE_DIM + j] = nominal_trajectory(j, i);
        actual_feedback_trajectory_save[t * num_timesteps * DYN::STATE_DIM + i * DYN::STATE_DIM + j] =
            feedback_state_trajectory(j, i);
      }
    }

    // Get the open loop control
    DYN::control_array current_control = nominal_control.col(0);

    // Apply the feedback given the current state
    current_control += vanilla_controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);

    // Propagate the state forward
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    vanilla_controller.slideControlSequence(1);
  }

  cnpy::npy_save("vanilla_large_track_actual.npy", nominal_trajectory_save.data(),
                 { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
  cnpy::npy_save("vanilla_large_track_feedback.npy", nominal_trajectory_save.data(),
                 { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
  // std::cout << "Number of times constraints were violated: " << fail_count << std::endl;
}

TEST_F(DoubleIntegratorTracking, TubeMPPILargeVariance)
{
  // Noise enters the system during the "true" state propagation. In this case the noise is nominal
  model.setStateVariance(100);
  auto fb_params = fb_controller.getParams();
  /**
   * Q =
   * [500, 0, 0, 0
   *  0, 500, 0, 0
   *  0, 0, 100, 0
   *  0, 0, 0, 100]
   */
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  /**
   * Qf = I
   */
  fb_params.Q_f = FB_CONTROLLER::square_state_matrix::Identity();
  /**
   * R = I
   */
  fb_params.R = FB_CONTROLLER::square_control_matrix::Identity();
  fb_controller.setParams(fb_params);

  // To pass it should be lambda = 4, vel desired = 2, vel cost = 1 crash cost 1000, nom threshold 20

  std::vector<float> actual_trajectory_save(num_timesteps * total_time_horizon * DYN::STATE_DIM);
  std::vector<float> nominal_trajectory_save(num_timesteps * total_time_horizon * DYN::STATE_DIM);
  // std::vector<float> ancillary_trajectory_save(num_timesteps*total_time_horizon*DYN::STATE_DIM);
  std::vector<float> feedback_trajectory_save(num_timesteps * total_time_horizon * DYN::STATE_DIM);

  // Initialize the tube MPPI controller
  auto controller = TUBE_CONTROLLER(&model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha, num_timesteps,
                                    num_rollouts);
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(50, 1, 1);
  // controller_params.seed_ = 42;
  controller.setParams(controller_params);

  controller.setNominalThreshold(100);

  int fail_count = 0;
  int crash_status[1] = { 0 };

  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t)
  {
    //     Print the system state
    if (t % 100 == 0)
    {
      float current_cost = cost.computeStateCost(x, 1, crash_status);
      printf("Current Time: %f    ", t * dt);
      printf("Current State Cost: %f    ", current_cost);
      model.printState(x.data());
      auto free_energy_stats = controller.getFreeEnergyStatistics();
      std::cout << "Real    FE [mean, variance]: [" << free_energy_stats.real_sys.freeEnergyMean << ", "
                << free_energy_stats.real_sys.freeEnergyVariance << "]" << std::endl;
      std::cout << "Nominal FE [mean, variance]: [" << free_energy_stats.nominal_sys.freeEnergyMean << ", "
                << free_energy_stats.nominal_sys.freeEnergyVariance << "]" << std::endl;
      std::cout << "Algorithm Health Normalizer: [" << controller.getNormalizerPercent() << "]\n" << std::endl;
    }

    if (cost.computeStateCost(x, t, crash_status) > 1000)
    {
      fail_count++;
      crash_status[0] = 0;
    }

    if (tubeFailure(x.data()))
    {
      float current_cost = cost.computeStateCost(x, 1, crash_status);
      printf("Current Time: %f    ", t * dt);
      printf("Current State Cost: %f    ", current_cost);
      model.printState(x.data());
      auto free_energy_stats = controller.getFreeEnergyStatistics();
      std::cout << "Real    FE [mean, variance]: [" << free_energy_stats.real_sys.freeEnergyMean << ", "
                << free_energy_stats.real_sys.freeEnergyVariance << "]" << std::endl;
      std::cout << "Nominal FE [mean, variance]: [" << free_energy_stats.nominal_sys.freeEnergyMean << ", "
                << free_energy_stats.nominal_sys.freeEnergyVariance << "]" << std::endl;
      std::cout << "Algorithm Health Normalizer: [" << controller.getNormalizerPercent() << "]\n" << std::endl;
      cnpy::npy_save("tube_large_actual.npy", actual_trajectory_save.data(),
                     { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
      // cnpy::npy_save("tube_ancillary.npy", ancillary_trajectory_save.data(),
      //                {total_time_horizon, num_timesteps, DYN::STATE_DIM},"w");
      cnpy::npy_save("tube_large_nominal.npy", nominal_trajectory_save.data(),
                     { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
      cnpy::npy_save("tube_large_feedback.npy", feedback_trajectory_save.data(),
                     { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
      FAIL() << "Visualize the trajectories by running scripts/double_integrator/plot_DI_test_trajectories; "
                "the argument to this python file is the build directory of MPPI-Generic";
    }

    // Compute the control
    controller.computeControl(x, 1);

    // Save the trajectory from the nominal state
    auto nominal_trajectory = controller.getTargetStateSeq();
    auto actual_trajectory = controller.getActualStateSeq();

    // Get the feedback gains associated with the nominal state and control trajectory
    controller.computeFeedback(x);

    // Save the ancillary trajectory
    // auto ancillary_trajectory = controller.getAncillaryStateSeq();

    // Compute the propagated feedback trajectory
    controller.computeFeedbackPropagatedStateSeq();
    auto propagated_feedback_trajectory = controller.getFeedbackPropagatedStateSeq();

    for (int i = 0; i < num_timesteps; i++)
    {
      for (int j = 0; j < DYN::STATE_DIM; j++)
      {
        actual_trajectory_save[t * num_timesteps * DYN::STATE_DIM + i * DYN::STATE_DIM + j] = actual_trajectory(j, i);
        // ancillary_trajectory_save[t * num_timesteps * DYN::STATE_DIM +
        //                         i*DYN::STATE_DIM + j] = ancillary_trajectory(j, i);
        nominal_trajectory_save[t * num_timesteps * DYN::STATE_DIM + i * DYN::STATE_DIM + j] = nominal_trajectory(j, i);
        feedback_trajectory_save[t * num_timesteps * DYN::STATE_DIM + i * DYN::STATE_DIM + j] =
            propagated_feedback_trajectory(j, i);
      }
    }

    // Get the open loop control
    DYN::control_array current_control = controller.getControlSeq().col(0);

    // Apply the feedback given the current state
    current_control += controller.getFeedbackControl(x, controller.getTargetStateSeq().col(0), 0);

    // Propagate the state forward
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    controller.slideControlSequence(1);
  }

  cnpy::npy_save("tube_large_actual.npy", actual_trajectory_save.data(),
                 { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
  // cnpy::npy_save("tube_ancillary.npy",ancillary_trajectory_save.data(),
  //                {total_time_horizon, num_timesteps, DYN::STATE_DIM},"w");
  cnpy::npy_save("tube_large_nominal.npy", nominal_trajectory_save.data(),
                 { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
  cnpy::npy_save("tube_large_feedback.npy", feedback_trajectory_save.data(),
                 { total_time_horizon, num_timesteps, DYN::STATE_DIM }, "w");
}
