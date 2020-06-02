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
         DoubleIntegratorDynamics, DoubleIntegratorCircleCost, 100, 512, 64, 8>{
 public:
  TestRobust(DoubleIntegratorDynamics *model,
          DoubleIntegratorCircleCost *cost,
          float dt, int max_iter, float gamma,
          const Eigen::Ref<const control_array>& control_std_dev,
          int num_timesteps,
          const Eigen::Ref<const control_trajectory>& init_control_traj,
          cudaStream_t stream) :
  RobustMPPIController(model, cost, dt,  max_iter,  gamma, control_std_dev, num_timesteps, init_control_traj, stream) {}


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
    candidate_trajectory_costs = traj_costs_in;
    return computeCandidateBaseline();
  }

  int getComputeBestIndex(const Eigen::Ref<const Eigen::MatrixXf>& traj_costs_in) {
    candidate_trajectory_costs = traj_costs_in;
    computeBestIndex();
    return best_index_;
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
    test_controller = new TestRobust(model, cost, dt, 3, gamma, control_std_dev, 100, init_control_traj, 0);
  }

  void TearDown() override {
    delete model;
    delete cost;
    delete test_controller;
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
    test_controller = new TestRobust(model, cost, dt, 3, gamma, control_variance, 100, init_control_traj, 0);

    // Set the size of the trajectory costs function
    trajectory_costs.resize(num_samples*num_candidates, 1);

    // Fill the trajectory costs with random costs
    trajectory_costs = Eigen::MatrixXf::Random(num_samples*num_candidates, 1);
  }

  void TearDown() override {
    delete model;
    delete cost;
    delete test_controller;
  }

  dynamics* model;
  cost_function* cost;
  TestRobust* test_controller;
  dynamics::control_array control_variance;
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

