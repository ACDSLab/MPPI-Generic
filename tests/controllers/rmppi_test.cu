#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <cnpy.h>
#include <random> // Used to generate random noise for control trajectories



/******************************************************************************
 * Test class for RobustControllerPrivateMethods
 ******************************************************************************/
class TestRobust: public RobustMPPIController<
         DoubleIntegratorDynamics, DoubleIntegratorCircleCost, 100, 2048, 64, 8>{
 public:
  TestRobust(DoubleIntegratorDynamics *model,
          DoubleIntegratorCircleCost *cost,
          float dt, int max_iter, float gamma,
          const Eigen::Ref<const StateCostWeight>& Q,
          const Eigen::Ref<const Hessian>& Qf,
          const Eigen::Ref<const ControlCostWeight>& R,
          const Eigen::Ref<const control_array>& control_std_dev,
          int num_timesteps,
          const Eigen::Ref<const control_trajectory>& init_control_traj,
          cudaStream_t stream) :
  RobustMPPIController(model, cost, dt,  max_iter,  gamma, Q, Qf, R, control_std_dev, num_timesteps, init_control_traj, 1, stream) {};


  // Test to make sure that its nonzero
  // Test to make sure that cuda memory is allocated
  NominalCandidateVector getCandidates(
          const Eigen::Ref<const state_array>& nominal_x_k,
          const Eigen::Ref<const state_array>& nominal_x_kp1,
          const Eigen::Ref<const state_array>& real_x_kp1) {
    getInitNominalStateCandidates(nominal_x_k, nominal_x_kp1, real_x_kp1);
    return candidate_nominal_states_;
  };

  Eigen::MatrixXf getWeights() {
    return line_search_weights_;
  };

  void updateCandidates(int value) {
    updateNumCandidates(value);
  }

  bool getCudaMemStatus() {
    return importance_sampling_cuda_mem_init_;
  }

  void deallocateNSCMemory() {
    deallocateNominalStateCandidateMemory();
  }

  void resetNSCMemory() {
    resetCandidateCudaMem();
  }

  Eigen::MatrixXi getStrideIS(int stride) {
    computeImportanceSamplerStride(stride);
    return importance_sampler_strides_;
  }

  float getComputeCandidateBaseline(const Eigen::Ref<const Eigen::MatrixXf>& traj_costs_in) {
    candidate_trajectory_costs_ = traj_costs_in;
    return computeCandidateBaseline();
  }

  int getComputeBestIndex(const Eigen::Ref<const Eigen::MatrixXf>& traj_costs_in) {
    candidate_trajectory_costs_ = traj_costs_in;
    computeBestIndex();
    return best_index_;
  }

  state_array getNominalStateFromOptimization(const Eigen::Ref<const state_array>& nominal_x_k,
                                              const Eigen::Ref<const state_array>& nominal_x_kp1,
                                              const Eigen::Ref<const state_array>& real_x_kp1,
                                              bool nominal_state_init) {
    nominal_state_trajectory_.col(0) = nominal_x_k;
    nominal_state_trajectory_.col(1) = nominal_x_kp1;
    nominal_state_init_ = nominal_state_init;
    computeNominalStateAndStride(real_x_kp1, 1); // Default the stride to 1
    return nominal_state_;
  }

  std::vector<float> getFeedbackGainVector() {
    return this->feedback_gain_vector_;
  }

 };

// Text fixture for nominal state selection
class RMPPINominalStateCandidates : public ::testing::Test {
public:
  using dynamics = DoubleIntegratorDynamics;
  using cost_function = DoubleIntegratorCircleCost;

protected:
  void SetUp() override {
    model = new dynamics(10);  // Initialize the double integrator dynamics
    cost = new cost_function;  // Initialize the cost function
    control_std_dev << 0.0001, 0.0001;
    init_control_traj.setZero();
    // Q, Qf, R
    Eigen::Matrix<float, dynamics::STATE_DIM, dynamics::STATE_DIM> Q, Qf;
    Eigen::Matrix<float, dynamics::CONTROL_DIM, dynamics::CONTROL_DIM> R;
    Q.setIdentity();
    Qf.setIdentity();
    R.setIdentity();
    test_controller = new TestRobust(model, cost, dt, 3, gamma, Q, Qf, R, control_std_dev, 100, init_control_traj, 0);
  }

  void TearDown() override {
    delete test_controller;
    delete model;
    delete cost;
  }

  dynamics* model;
  cost_function* cost;
  TestRobust* test_controller;
  dynamics::control_array control_std_dev;
  TestRobust::control_trajectory init_control_traj;
  float dt = 0.01;
  float gamma = 0.5;
};

TEST_F(RMPPINominalStateCandidates, UpdateNumCandidates_LessThan3) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(test_controller->updateCandidates(1) , "ERROR: number of candidates must be greater or equal to 3\n");
}

TEST_F(RMPPINominalStateCandidates, UpdateNumCandidates_Negative) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(test_controller->updateCandidates(-1) , "ERROR: number of candidates must be greater or equal to 3\n");
}

TEST_F(RMPPINominalStateCandidates, UpdateNumCandidates_Even) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(test_controller->updateCandidates(4) , "ERROR: number of candidates must be odd\n");
}

TEST_F(RMPPINominalStateCandidates, UpdateNumCandidates_TooManyCandidates) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(test_controller->updateCandidates(99), "cannot exceed");
}

TEST_F(RMPPINominalStateCandidates, CandidateVectorSizeNonzero) {
  dynamics::state_array x_star_k, x_star_kp1, x_kp1;
  x_star_k << -4 , 0, 0, 0;
  x_star_kp1 << 4, 0, 0, 0;
  x_kp1 << 4, 4, 0, 0;
  auto candidates = test_controller->getCandidates(x_star_k, x_star_kp1, x_kp1);
  ASSERT_TRUE(candidates.size() > 0);
}

TEST_F(RMPPINominalStateCandidates, CudaMemoryInitialized) {
  ASSERT_TRUE(test_controller->getCudaMemStatus());
}

TEST_F(RMPPINominalStateCandidates, CudaMemoryRemoved) {
  test_controller->deallocateNSCMemory();
  ASSERT_FALSE(test_controller->getCudaMemStatus());
}

TEST_F(RMPPINominalStateCandidates, CudaMemoryReset) {
  test_controller->deallocateNSCMemory(); // Remove memory
  ASSERT_FALSE(test_controller->getCudaMemStatus());
  test_controller->resetNSCMemory(); // Should allocate
  ASSERT_TRUE(test_controller->getCudaMemStatus());
  test_controller->resetNSCMemory(); // Should remove then allocate again
  ASSERT_TRUE(test_controller->getCudaMemStatus());
}

TEST_F(RMPPINominalStateCandidates, LineSearchWeights_9) {
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

TEST_F(RMPPINominalStateCandidates, ImportanceSampler_Stride_2) {
  int stride = 2;
  test_controller->updateCandidates(9);
  Eigen::MatrixXi known_stride(1,9);
  known_stride << 0, 1, 1, 2, 2, 2, 2, 2, 2;
  auto compute_stride = test_controller->getStrideIS(stride);
  ASSERT_TRUE(known_stride == compute_stride)
    << "Known Stride: \n" << known_stride << "\nComputed Stride: \n" << compute_stride;
}

TEST_F(RMPPINominalStateCandidates, ImportanceSampler_Stride_4) {
  int stride = 4;
  test_controller->updateCandidates(9);
  Eigen::MatrixXi known_stride(1,9);
  known_stride << 0, 1, 2, 3, 4, 4, 4, 4, 4;
  auto compute_stride = test_controller->getStrideIS(stride);
  ASSERT_TRUE(known_stride == compute_stride)
                        << "Known Stride: \n" << known_stride << "\nComputed Stride: \n" << compute_stride;
}

TEST_F(RMPPINominalStateCandidates, InitEvalSelection_Weights) {
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
  known_candidates << -4, -3, -2, -1, 0, 1, 2, 3, 4,
                       0,  1,  2,  3, 4, 4, 4, 4, 4,
                       0,  0,  0,  0, 0, 0, 0, 0, 0,
                       0,  0,  0,  0, 0, 0, 0, 0, 0;

  // Create the controller so we can test functions from it -> this should be a fixture
  auto candidates = test_controller->getCandidates(x_star_k, x_star_kp1, x_kp1);

  for (int i = 0; i < num_candidates; ++i) {
    ASSERT_TRUE(candidates[i] == known_candidates.col(i))
      << "Index: " << i
      << "\nKnown Point: \n" << known_candidates.col(i)
      << "\nComputed Point: \n" << candidates[i];
  }

}

class RMPPINominalStateSelection : public ::testing::Test {
public:
  using dynamics = DoubleIntegratorDynamics;
  using cost_function = DoubleIntegratorCircleCost;
  int num_samples = 64;
  int num_candidates = 9;
  Eigen::MatrixXf trajectory_costs;
protected:
  void SetUp() override {
    model = new dynamics(10);  // Initialize the double integrator dynamics
    cost = new cost_function;  // Initialize the cost function
    control_std_dev << 0.0001, 0.0001;
    init_control_traj.setZero();

    // Q, Qf, R
    Eigen::Matrix<float, dynamics::STATE_DIM, dynamics::STATE_DIM> Q, Qf;
    Eigen::Matrix<float, dynamics::CONTROL_DIM, dynamics::CONTROL_DIM> R;
    Q.setIdentity();
    Qf.setIdentity();
    R.setIdentity();

    test_controller = new TestRobust(model, cost, dt, 3, gamma, Q, Qf, R, control_std_dev, 100, init_control_traj, 0);
    test_controller->value_func_threshold_ = 10.0;

    // Set the size of the trajectory costs function
    trajectory_costs.resize(num_samples*num_candidates, 1);

    // Fill the trajectory costs with random costs
    trajectory_costs = Eigen::MatrixXf::Random(num_samples*num_candidates, 1);
  }

  void TearDown() override {
    delete test_controller;
    delete cost;
    delete model;
  }

  dynamics* model;
  cost_function* cost;
  TestRobust* test_controller;
  dynamics::control_array control_std_dev;
  TestRobust::control_trajectory init_control_traj;
  float dt = 0.01;
  float gamma = 0.5;

};

TEST_F(RMPPINominalStateSelection, GetCandidateBaseline) {
  // Compute baseline
  float baseline = trajectory_costs(0);
  for (int i = 0; i < num_samples; i++){
    if (trajectory_costs(i) < baseline){
      baseline = trajectory_costs(i);
    }
  }

  //
  float compute_baseline = test_controller->getComputeCandidateBaseline(trajectory_costs);

  ASSERT_FLOAT_EQ(baseline, compute_baseline);
}

TEST_F(RMPPINominalStateSelection, ComputeBestCandidate) {
  float baseline = trajectory_costs(0);
  for (int i = 0; i < num_samples; i++){
    if (trajectory_costs(i) < baseline){
      baseline = trajectory_costs(i);
    }
  }
  float value_func_threshold_ = 1000.0;

  Eigen::MatrixXf candidate_free_energy;
  candidate_free_energy.resize(num_candidates, 1);
  candidate_free_energy.setZero();
  // Should probably be in a cuda kernel? Will have to profile.
  for (int i = 0; i < num_candidates; i++){
    for (int j = 0; j < num_samples; j++){
      candidate_free_energy(i) += expf(-gamma*(trajectory_costs(i*num_samples + j) - baseline));
    }
  }
  for (int i = 0; i < num_candidates; i++){
    candidate_free_energy(i) /= (1.0*num_samples);
  }

  for (int i = 0; i < num_candidates; i++){
    candidate_free_energy(i) = -1.0/gamma*logf(candidate_free_energy(i)) + baseline;
  }

  //Now get the closest initial condition that is above the threshold.
  int bestIdx = 0;
  for (int i = 1; i < num_candidates; i++){
    if (candidate_free_energy(i) < value_func_threshold_){
      bestIdx = i;
    }
  }

  int best_index_compute = test_controller->getComputeBestIndex(trajectory_costs);

  ASSERT_EQ(bestIdx, best_index_compute);
}

TEST_F(RMPPINominalStateSelection, ComputeNominalStateAndStride_InitFalse) {
  // Set the number of candidates to 3
  const int num_candidates = 3;
  test_controller->updateCandidates(num_candidates);

  // We know that the cost penalizes any trajectory that exists our donut which
  // is centered around the origin with radius 2. Set the 3 relevant points of
  // the system such that x_k_star (the previous nominal state) is the best
  // free energy candidate.
  dynamics::state_array nominal_x_k, nominal_x_kp1, real_x_kp1;
  nominal_x_k << 2, 0, 0, 0;
  nominal_x_kp1 << 5, 0, 0, 0;
  real_x_kp1 << 5, 0, 0, 0;

  bool nominal_state_init = false;

  auto nominal_state =  test_controller->getNominalStateFromOptimization(nominal_x_k,
          nominal_x_kp1, real_x_kp1, nominal_state_init);

  // Since nominal state init is false, this should be the real state
  ASSERT_EQ(real_x_kp1, nominal_state) << "\nExpected state << \n" << real_x_kp1 << "\nComputed state: \n" << nominal_state;
}

TEST_F(RMPPINominalStateSelection, ComputeNominalStateAndStride_PreviousNominal) {
  // Set the number of candidates to 3
  const int num_candidates = 3;
  test_controller->updateCandidates(num_candidates);

  // We know that the cost penalizes any trajectory that exists our donut which
  // is centered around the origin with radius 2. Set the 3 relevant points of
  // the system such that x_k_star (the previous nominal state) is the best
  // free energy candidate.
  dynamics::state_array nominal_x_k, nominal_x_kp1, real_x_kp1;
  nominal_x_k << 2, 0, 0, 0;
  nominal_x_kp1 << 15, 0, 0, 0;
  real_x_kp1 << 15, 0, 0, 0;

  bool nominal_state_init = true;

  auto nominal_state =  test_controller->getNominalStateFromOptimization(nominal_x_k,
                                                                         nominal_x_kp1, real_x_kp1, nominal_state_init);

  // Since nominal state init is false, this should be the real state
  ASSERT_EQ(nominal_x_k, nominal_state) << "\nExpected state << \n" << nominal_x_k << "\nComputed state: \n" << nominal_state;
}

TEST_F(RMPPINominalStateSelection, ComputeNominalStateAndStride_CurrentNominal) {
  // Set the number of candidates to 3
  const int num_candidates = 3;
  test_controller->updateCandidates(num_candidates);

  // We know that the cost penalizes any trajectory that exists our donut which
  // is centered around the origin with radius 2. Set the 3 relevant points of
  // the system such that x_k_star (the previous nominal state) is the best
  // free energy candidate.
  dynamics::state_array nominal_x_k, nominal_x_kp1, real_x_kp1;
  nominal_x_k << -100, 0, 0, 0;
  nominal_x_kp1 << 2, 0, 0, 0;
  real_x_kp1 << 100, 0, 0, 0;

  bool nominal_state_init = true;

  auto nominal_state =  test_controller->getNominalStateFromOptimization(nominal_x_k,
                                                                         nominal_x_kp1, real_x_kp1, nominal_state_init);

  // Since nominal state init is false, this should be the real state
  ASSERT_EQ(nominal_x_kp1, nominal_state) << "\nExpected state << \n" << nominal_x_kp1 << "\nComputed state: \n" << nominal_state;
}

TEST_F(RMPPINominalStateSelection, ComputeNominalStateAndStride_CurrentReal) {
  // Set the number of candidates to 3
  const int num_candidates = 3;
  test_controller->updateCandidates(num_candidates);

  // We know that the cost penalizes any trajectory that exists our donut which
  // is centered around the origin with radius 2. Set the 3 relevant points of
  // the system such that x_k_star (the previous nominal state) is the best
  // free energy candidate.
  dynamics::state_array nominal_x_k, nominal_x_kp1, real_x_kp1;
  nominal_x_k << -100, 0, 0, 0;
  nominal_x_kp1 << -100, 0, 0, 0;
  real_x_kp1 << 2, 0, 0, 0;

  bool nominal_state_init = true;

  auto nominal_state =  test_controller->getNominalStateFromOptimization(nominal_x_k,
                                                                         nominal_x_kp1, real_x_kp1, nominal_state_init);

  // Since nominal state init is false, this should be the real state
  ASSERT_EQ(real_x_kp1, nominal_state) << "\nExpected state << \n" << real_x_kp1 << "\nComputed state: \n" << nominal_state;
}

TEST_F(RMPPINominalStateSelection, FeedbackGainInternalStorage) {
  dynamics::state_array x;
  x << 2, 0, 0, 0;
  int stride = 1;
  test_controller->updateImportanceSampler(x, stride);
  test_controller->updateImportanceSampler(x, stride);

  std::vector<float> feedback_gain_vector = test_controller->getFeedbackGainVector();

  auto feedback_gain_eigen_aligned = test_controller->getFeedbackGains();


  for (size_t i = 0; i < feedback_gain_eigen_aligned.size(); i++) {
    int i_index = i * dynamics::STATE_DIM * dynamics::CONTROL_DIM;
    for (size_t j = 0; j < dynamics::CONTROL_DIM * dynamics::STATE_DIM; j++) {
      ASSERT_FLOAT_EQ(feedback_gain_vector[i_index + j] , feedback_gain_eigen_aligned[i].data()[j]);
    }
  }
}