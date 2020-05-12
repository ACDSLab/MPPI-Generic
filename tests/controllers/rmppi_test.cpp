#include <gtest/gtest.h>
#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>

TEST(RMPPITest, CPURolloutKernel) {
  DoubleIntegratorDynamics model;
  DoubleIntegratorCircleCost cost;

  const int state_dim = DoubleIntegratorDynamics::STATE_DIM;
  const int control_dim = DoubleIntegratorDynamics::CONTROL_DIM;
  typedef Eigen::Matrix<float, control_dim, state_dim> K_M;

  float dt = 0.01;
  int max_iter = 10;
  float gamma = 0.5;
  const int num_timesteps = 100;
  const int num_rollouts = 5;

  float x[num_rollouts * state_dim * 2];
  float x_dot[num_rollouts * state_dim * 2];
  float u[num_rollouts * control_dim * 2];
  float du[num_rollouts * control_dim * 2];
  float sigma_u[control_dim]; // variance to sample noise from
  float fb_u[num_rollouts * control_dim];

  // Generate control noise
  float noisey_sample[num_rollouts * num_timesteps * control_dim * 2];
  // Initial control trajectory
  float u_traj[num_rollouts * num_timesteps * control_dim * 2];

  VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost, 100, 512, 64, 8>::feedback_gain_trajectory feedback_gains;
  for (int i = 0; i < 10; i++) {
    feedback_gains.push_back(K_M::Random());
  }

  float feedback_array[num_rollouts * num_timesteps * control_dim * state_dim * 2];
  for (size_t i = 0; i < feedback_gains.size(); i++) {
    std::cout << "Matrix " << i << ":\n";
    std::cout << feedback_gains[i] << std::endl;
  }

  for (int traj_i = 0; traj_i < num_rollouts; traj_i++)  {
    float cost_real_w_tracking = 0; // S^(V, x_0, x*_0) in Grady Thesis (8.24)
    float cost_real = 0; // S(V, x_0) with knowledge of tracking controller
    float cost_nom = 0; // S(V, x*_0)

    int traj_index = traj_i * num_rollouts;

     // Get all relevant values at time t in rollout i
    DoubleIntegratorDynamics::state_array x_t_nom;
    DoubleIntegratorDynamics::state_array x_t_act;
    // Eigen::Map<DoubleIntegratorDynamics::state_array> x_t_act(x + traj_index * state_dim);
    for (int state_i = 0; state_i < state_dim; state_i++) {
      x_t_act(state_i, 0) = x[traj_index * state_dim + state_i];
      x_t_nom(state_i, 0) = x[(traj_index + num_rollouts) * state_dim + state_i];
    }

    for (int control_i = 0; control_i < control_dim; ) {

    }
    for (int t = 0; t < num_timesteps - 1; t++){
      // Controls are read only so I can use Eigen::Map
      Eigen::Map<DoubleIntegratorDynamics::control_array>
        u_t(u_traj + (traj_index + num_timesteps) * control_dim); // trajectory u at time t
      Eigen::Map<DoubleIntegratorDynamics::control_array>
        eps_t(noisey_sample + (traj_index + num_timesteps) * control_dim); // Noise at time t
      Eigen::Matrix<float, control_dim, state_dim> feedback_gains_t; // Feedback gains at time t

      // Create newly calculated values at time t in rollout i
      DoubleIntegratorDynamics::state_array x_dot_t_nom;
      DoubleIntegratorDynamics::state_array x_dot_t_act;
      DoubleIntegratorDynamics::control_array u_nom = u_t + eps_t;
      DoubleIntegratorDynamics::control_array fb_u_t = feedback_gains_t * (x_t_nom - x_t_act);
      DoubleIntegratorDynamics::control_array u_act = u_nom + fb_u_t;
      // Dyanamics Update
      model.computeStateDeriv(x_t_nom, u_nom, x_dot_t_nom);
      model.computeStateDeriv(x_t_act, u_act, x_dot_t_act);

      // Cost update
      // cost_real_w_tracking += cost->computeRunningCost(x_t_act.data(), u_t.data(), u_nom_t.data(),);

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
