#include <gtest/gtest.h>
#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>
#include <random> // Used to generate random noise for control trajectories

TEST(RMPPITest, CPURolloutKernel) {
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  DYN model;
  COST cost;

  const int state_dim = DYN::STATE_DIM;
  const int control_dim = DYN::CONTROL_DIM;

  float dt = 0.01;
  int max_iter = 10;
  float lambda = 0.5;
  const int num_timesteps = 7;
  const int num_rollouts = 5;

  // float x[num_rollouts * state_dim * 2];
  // float x_dot[num_rollouts * state_dim * 2];
  // float u[num_rollouts * control_dim * 2];
  // float du[num_rollouts * control_dim * 2];
  float sigma_u[control_dim] = {0.5, 0.4}; // variance to sample noise from
  // float fb_u[num_rollouts * control_dim * state_dim];

  DYN::state_array x_init_act;
  x_init_act << 0, 0, 0, 0;
  DYN::state_array x_init_nom;

  // Generate control noise
  float sampled_noise[num_rollouts * num_timesteps * control_dim * 2];
  std::mt19937 rng_gen;
  std::vector<std::normal_distribution<float>> control_dist;
  for (int i = 0; i < control_dim; i++) {
    control_dist.push_back(std::normal_distribution<float>(0, sigma_u[i]));
  }

  for (int n = 0; n < num_rollouts; n++) {
    int n_ind = n * num_timesteps * control_dim;
    for (int t = 0; t < num_timesteps; t++) {
      int t_ind = t * control_dim;
      for (int j = 0; j < control_dim; j++) {
        sampled_noise[n_ind + t_ind + j] = control_dist[j](rng_gen);
      }
    }
  }
  // TODO: Figure out nonzero Initial control trajectory
  float u_traj[num_rollouts * num_timesteps * control_dim * 2] = {0};

  // TODO: fill the variance in with more reasonable numbers
  COST::control_matrix cost_variance = COST::control_matrix::Identity();

  // TODO: Generate feedback gain trajectories
  VanillaMPPIController<DYN, COST, 100, 512, 64, 8>::feedback_gain_trajectory feedback_gains;
  for (int i = 0; i < num_timesteps; i++) {
    feedback_gains.push_back(DYN::feedback_matrix::Random());
  }

  // Copy Feedback Gains into an array
  float feedback_array[num_timesteps * control_dim * state_dim];
  for (size_t i = 0; i < feedback_gains.size(); i++) {
    // std::cout << "Matrix " << i << ":\n";
    // std::cout << feedback_gains[i] << std::endl;
    int i_index = i * control_dim * state_dim;

    for (size_t j = 0; j < control_dim * state_dim; j++) {
      feedback_array[i_index + j] = feedback_gains[i].data()[j];
    }
  }

  for (int traj_i = 0; traj_i < num_rollouts; traj_i++)  {
    float cost_real_w_tracking = 0; // S^(V, x_0, x*_0) in Grady Thesis (8.24)
    float total_cost_real = 0; // S(V, x_0) with knowledge of tracking controller
    float state_cost_nom = 0; // S(V, x*_0)

    int traj_index = traj_i * num_rollouts;

     // Get all relevant values at time t in rollout i
    DYN::state_array x_t_nom = x_init_nom;
    DYN::state_array x_t_act = x_init_act;
    // Eigen::Map<DYN::state_array> x_t_act(x + traj_index * state_dim);
    // for (int state_i = 0; state_i < state_dim; state_i++) {
    //   x_t_act(state_i, 0) = x[traj_index * state_dim + state_i];
    //   x_t_nom(state_i, 0) = x[(traj_index + num_rollouts) * state_dim + state_i];
    // }

    for (int t = 0; t < num_timesteps - 1; t++){
      // Controls are read only so I can use Eigen::Map<const...>
      Eigen::Map<const DYN::control_array>
        u_t(u_traj + (traj_index * num_timesteps + t) * control_dim); // trajectory u at time t
      Eigen::Map<const DYN::control_array>
        eps_t(sampled_noise + (traj_index * num_timesteps + t) * control_dim); // Noise at time t
      Eigen::Map<const DYN::feedback_matrix>
        feedback_gains_t(feedback_array + t * control_dim * state_dim); // Feedback gains at time t
      // if (traj_i == 0) {
      //   std::cout << "feedback_gains_t " << traj_i << ", " << t << "s:\n" << feedback_gains_t << std::endl;
      // }


      // Create newly calculated values at time t in rollout i
      DYN::state_array x_dot_t_nom;
      DYN::state_array x_dot_t_act;
      DYN::control_array u_nom = u_t + eps_t;
      DYN::control_array fb_u_t = feedback_gains_t * (x_t_nom - x_t_act);
      DYN::control_array u_act = u_nom + fb_u_t;

      // Cost update
      DYN::control_array zero_u = DYN::control_array::Zero();
      state_cost_nom += cost.computeStateCost(x_t_nom);
      float state_cost_act = cost.computeStateCost(x_t_act);
      cost_real_w_tracking +=  state_cost_act +
        cost.computeFeedbackCost(zero_u, zero_u, fb_u_t, cost_variance, lambda);

      total_cost_real += state_cost_act +
        cost.computeLikelihoodRatioCost(u_t + fb_u_t, eps_t, cost_variance, lambda);

      // Dyanamics Update
      model.computeStateDeriv(x_t_nom, u_nom, x_dot_t_nom);
      model.computeStateDeriv(x_t_act, u_act, x_dot_t_act);

      model.updateState(x_t_act, x_dot_t_act, dt);
      model.updateState(x_t_nom, x_dot_t_nom, dt);
    }
    // cost_real_w_tracking += TERMINAL_COST(x_t_act);
    // state_cost_nom += TERMINAL_COST(x_t_nom);
    // total_cost_real += += TERMINAL_COST(x_t_act);
    // TODO Choose alpha better
    float alpha = 0.5;
    float cost_nom = 0.5 * state_cost_nom + 0.5 * std::max(std::min(cost_real_w_tracking, alpha), state_cost_nom);
    // std::cout << "for loop problems, I feel bad for you son" << std::endl;
    for (int t = 0; t < num_timesteps - 1; t++) {
      Eigen::Map<DYN::control_array>
        u_t(u_traj + (traj_index + num_timesteps) * control_dim); // trajectory u at time t
      Eigen::Map<DYN::control_array>
        eps_t(sampled_noise + (traj_index + num_timesteps) * control_dim); // Noise at time t
      cost_nom += cost.computeLikelihoodRatioCost(u_t, eps_t, cost_variance);
    }
  }
}

/******************************************************************************
 * Test class for RobustControllerPrivateMethods
 ******************************************************************************/
class TestRobust: public RobustMPPIController<
         DoubleIntegratorDynamics, DoubleIntegratorCircleCost, 100, 512, 64, 8>{
 public:
  TestRobust(DoubleIntegratorDynamics *model, DoubleIntegratorCircleCost *cost) :
  RobustMPPIController(model, cost) {}


  // Test to make sure that its nonzero
  // Test to make sure that cuda memory is allocated
  auto getCandidates(
          const Eigen::Ref<const state_array>& nominal_x_k,
          const Eigen::Ref<const state_array>& nominal_x_kp1,
          const Eigen::Ref<const state_array>& real_x_kp1) {
    getInitNominalStateCandidates(nominal_x_k, nominal_x_kp1, real_x_kp1);
    return candidate_nominal_states;
  };

  auto getWeights() {
    return line_search_weights;
  };

  void updateCandidates(int value) {
    updateNumCandidates(value);
  }

  bool getCudaMemStatus() {
    return importance_sampling_cuda_mem_init;
  }

  void deallocateNSCMemory() {
    deallocateNominalStateCandidateMemory();
  }

  void resetNSCMemory() {
    resetCandidateCudaMem();
  }

  auto getStrideIS(int stride) {
    computeImportanceSamplerStride(stride);
    return importance_sampler_strides;
  }

 };

// Text fixture for nominal state selection
class RMPPINominalStateSelection : public ::testing::Test {
public:
  using dynamics = DoubleIntegratorDynamics;
  using cost_function = DoubleIntegratorCircleCost;

protected:
  void SetUp() override {
    model = new dynamics(10);  // Initialize the double integrator dynamics
    cost = new cost_function;  // Initialize the cost function
    test_controller = new TestRobust(model, cost);
  }

  void TearDown() override {
    delete model;
    delete cost;
    delete test_controller;
  }

  dynamics* model;
  cost_function* cost;
  TestRobust* test_controller;
};

TEST_F(RMPPINominalStateSelection, UpdateNumCandidates_LessThan3) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(test_controller->updateCandidates(1) , "ERROR: number of candidates must be greater or equal to 3\n");
}

TEST_F(RMPPINominalStateSelection, UpdateNumCandidates_Negative) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(test_controller->updateCandidates(-1) , "ERROR: number of candidates must be greater or equal to 3\n");
}

TEST_F(RMPPINominalStateSelection, UpdateNumCandidates_Even) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(test_controller->updateCandidates(4) , "ERROR: number of candidates must be odd\n");
}

TEST_F(RMPPINominalStateSelection, CandidateVectorSizeNonzero) {
  dynamics::state_array x_star_k, x_star_kp1, x_kp1;
  x_star_k << -4 , 0, 0, 0;
  x_star_kp1 << 4, 0, 0, 0;
  x_kp1 << 4, 4, 0, 0;
  auto candidates = test_controller->getCandidates(x_star_k, x_star_kp1, x_kp1);
  ASSERT_TRUE(candidates.size() > 0);
}

TEST_F(RMPPINominalStateSelection, CudaMemoryInitialized) {
  ASSERT_TRUE(test_controller->getCudaMemStatus());
}

TEST_F(RMPPINominalStateSelection, CudaMemoryRemoved) {
  test_controller->deallocateNSCMemory();
  ASSERT_FALSE(test_controller->getCudaMemStatus());
}

TEST_F(RMPPINominalStateSelection, CudaMemoryReset) {
  test_controller->deallocateNSCMemory(); // Remove memory
  ASSERT_FALSE(test_controller->getCudaMemStatus());
  test_controller->resetNSCMemory(); // Should allocate
  ASSERT_TRUE(test_controller->getCudaMemStatus());
  test_controller->resetNSCMemory(); // Should remove then allocate again
  ASSERT_TRUE(test_controller->getCudaMemStatus());
}

TEST_F(RMPPINominalStateSelection, LineSearchWeights_9) {
  test_controller->updateCandidates(9);
  auto controller_weights = test_controller->getWeights();
  Eigen::MatrixXf known_weights(3,9);
  known_weights << 1, 3.0/4, 1.0/2, 1.0/4, 0, 0, 0, 0, 0,
                 0, 1.0/4, 1.0/2, 3.0/4, 1, 3.0/4, 1.0/2, 1.0/4, 0,
                 0, 0, 0, 0, 0, 1.0/4, 1.0/2, 3.0/4, 1;

//  std::cout << controller_weights << std::endl;
//  std::cout << known_weights << std::endl;
  ASSERT_TRUE(controller_weights == known_weights)
  << "Known Weights: \n" << known_weights << "\nComputed Weights: \n" << controller_weights;
}

TEST_F(RMPPINominalStateSelection, ImportanceSampler_Stride_2) {
  int stride = 2;
  test_controller->updateCandidates(9);
  Eigen::MatrixXi known_stride(1,9);
  known_stride << 0, 1, 1, 2, 2, 2, 2, 2, 2;
  auto compute_stride = test_controller->getStrideIS(stride);
  ASSERT_TRUE(known_stride == compute_stride)
    << "Known Stride: \n" << known_stride << "\nComputed Stride: \n" << compute_stride;
}

TEST_F(RMPPINominalStateSelection, ImportanceSampler_Stride_4) {
  int stride = 4;
  test_controller->updateCandidates(9);
  Eigen::MatrixXi known_stride(1,9);
  known_stride << 0, 1, 2, 3, 4, 4, 4, 4, 4;
  auto compute_stride = test_controller->getStrideIS(stride);
  ASSERT_TRUE(known_stride == compute_stride)
                        << "Known Stride: \n" << known_stride << "\nComputed Stride: \n" << compute_stride;
}

TEST_F(RMPPINominalStateSelection, InitEvalSelection_Weights) {
  /*
   * This test will ensure that the line search process to select the
   * number of points for free energy evaluation is correct.
   */
  dynamics::state_array x_star_k, x_star_kp1, x_kp1;
  x_star_k << -4 , 0, 0, 0;
  x_star_kp1 << 0, 4, 0, 0;
  x_kp1 << 4, 4, 0, 0;

  const int num_candidates = 9;
  test_controller->updateCandidates(num_candidates);

  Eigen::Matrix<float, 4, num_candidates> known_candidates;
  known_candidates << -4 , -3, -2, -1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4, 4, 4,
                        0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // Create the controller so we can test functions from it -> this should be a fixture
  auto candidates = test_controller->getCandidates(x_star_k, x_star_kp1, x_kp1);

  for (int i = 0; i < num_candidates; ++i) {
    ASSERT_TRUE(candidates[i] == known_candidates.col(i))
      << "Index: " << i
      << "\nKnown Point: \n" << known_candidates.col(i)
      << "\nComputed Point: \n" << candidates[i];
  }

}

TEST_F(RMPPINominalStateSelection, InitEvalRollout) {
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
  std::cout << cost_vector << std::endl;
}
