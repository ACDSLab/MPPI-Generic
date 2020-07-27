#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_robust_cost.cuh>
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
          float dt, int max_iter, float lambda, float alpha,
          float value_function_threshold,
          const Eigen::Ref<const StateCostWeight>& Q,
          const Eigen::Ref<const Hessian>& Qf,
          const Eigen::Ref<const ControlCostWeight>& R,
          const Eigen::Ref<const control_array>& control_std_dev,
          int num_timesteps,
          const Eigen::Ref<const control_trajectory>& init_control_traj,
          cudaStream_t stream) :
  RobustMPPIController(model, cost, dt,  max_iter, lambda, alpha, value_function_threshold, Q, Qf, R, control_std_dev, num_timesteps, init_control_traj, 9, 1, stream) {}

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
    dynamics::dfdx Q, Qf;  // Get a N x N matrix
    cost_function::control_matrix R; // Get a M x M matrix
    Q.setIdentity();
    Qf.setIdentity();
    R.setIdentity();
    test_controller = new TestRobust(model, cost, dt, 3, lambda, alpha, 1000.0, Q, Qf, R, control_std_dev, 100, init_control_traj, 0);
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
  float lambda = 0.5;
  float alpha = 0.01;
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
    dynamics::dfdx Q, Qf;  // Get a N x N matrix
    cost_function::control_matrix R; // Get a M x M matrix
    Q.setIdentity();
    Qf.setIdentity();
    R.setIdentity();

    test_controller = new TestRobust(model, cost, dt, 3, lambda, alpha, 1000.0, Q, Qf, R, control_std_dev, 100, init_control_traj, 0);

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
  float lambda = 0.5;
  float alpha = 0.01;

};

TEST_F(RMPPINominalStateSelection, GetCandidateBaseline) {
  // Compute baseline
  float baseline = trajectory_costs(0);
  for (int i = 0; i < trajectory_costs.size(); i++){
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
      candidate_free_energy(i) += expf(-1.0/lambda*(trajectory_costs(i*num_samples + j) - baseline));
    }
  }
  for (int i = 0; i < num_candidates; i++){
    candidate_free_energy(i) /= (1.0*num_samples);
  }

  for (int i = 0; i < num_candidates; i++){
    candidate_free_energy(i) = -lambda*logf(candidate_free_energy(i)) + baseline;
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
  // the system such that x_k_star (the nominal state) is the best
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
  test_controller->updateImportanceSamplingControl(x, stride);
  test_controller->updateImportanceSamplingControl(x, stride);

  std::vector<float> feedback_gain_vector = test_controller->getFeedbackGainVector();

  auto feedback_gain_eigen_aligned = test_controller->getFeedbackGains();


  for (size_t i = 0; i < feedback_gain_eigen_aligned.size(); i++) {
    int i_index = i * dynamics::STATE_DIM * dynamics::CONTROL_DIM;
    for (size_t j = 0; j < dynamics::CONTROL_DIM * dynamics::STATE_DIM; j++) {
      ASSERT_FLOAT_EQ(feedback_gain_vector[i_index + j] , feedback_gain_eigen_aligned[i].data()[j]);
    }
  }
}

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

TEST(RMPPITest, RobustMPPILargeVariance) {
  using DYNAMICS = DoubleIntegratorDynamics;
  using COST_T = DoubleIntegratorCircleCost;
  // Noise enters the system during the "true" state propagation. In this case the noise is nominal
  DYNAMICS model(100);  // Initialize the double integrator dynamics
  COST_T cost;  // Initialize the cost function
  float dt = 0.02; // Timestep of dynamics propagation
  int max_iter = 3; // Maximum running iterations of optimization
  float lambda = 0.25; // Learning rate parameter
  float alpha = 0.01;
  const int num_timesteps = 50;  // Optimization time horizon
  const int total_time_horizon = 5000;

  std::vector<float> actual_trajectory_save(num_timesteps*total_time_horizon*DYNAMICS::STATE_DIM);
  std::vector<float> nominal_trajectory_save(num_timesteps*total_time_horizon*DYNAMICS::STATE_DIM);
  std::vector<float> ancillary_trajectory_save(num_timesteps*total_time_horizon*DYNAMICS::STATE_DIM);


  // Set the initial state
  DYNAMICS::state_array x;
  x << 2, 0, 0, 1;

  DYNAMICS::state_array xdot;

  // control variance
  DYNAMICS::control_array control_var;
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
  Q = 500*DoubleIntegratorDynamics::dfdx::Identity();
  Q(2,2) = 100;
  Q(3,3) = 100;
  /**
   * R = I
   */
  R = 1*DoubleIntegratorCircleCost::control_matrix::Identity();

  /**
   * Qf = I
   */
  Qf = DoubleIntegratorDynamics::dfdx::Identity();

  // Value function threshold
  float value_function_threshold = 10.0;

  // DoubleIntegratorRobustCost cost2;
  // auto controller2 = RobustMPPIController<DYNAMICS, DoubleIntegratorRobustCost, num_timesteps,
  //         1024, 64, 8, 1>(&model, &cost2, dt, max_iter, gamma, value_function_threshold, Q, Qf, R, control_var);

  // Initialize the R MPPI controller
  auto controller = RobustMPPIController<DYNAMICS, COST_T, num_timesteps,
          1024, 64, 8, 1>(&model, &cost, dt, max_iter, lambda, alpha, value_function_threshold, Q, Qf, R, control_var);

  int fail_count = 0;

  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t) {
    // Print the system state
    if (t % 100 == 0) {
      printf("Current Time: %f    ", t * dt);
      model.printState(x.data());
      std::cout << "                          Candidate Free Energies: " << controller.getCandidateFreeEnergy().transpose() << std::endl;
    }

    if (cost.computeStateCost(x) > 1000) {
      fail_count++;
    }

    if (tubeFailure(x.data())) {
      cnpy::npy_save("robust_sc_large_actual.npy", actual_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
      cnpy::npy_save("robust_sc_ancillary.npy", ancillary_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
      cnpy::npy_save("robust_sc_large_nominal.npy",nominal_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
      printf("Current Time: %f    ", t * dt);
      model.printState(x.data());
      std::cout << "                          Candidate Free Energies: " << controller.getCandidateFreeEnergy().transpose() << std::endl;
      std::cout << "Tube failure!!" << std::endl;
      FAIL() << "Visualize the trajectories by running scripts/double_integrator/plot_DI_test_trajectories; "
                "the argument to this python file is the build directory of MPPI-Generic";
    }
    // Update the importance sampler
    controller.updateImportanceSamplingControl(x, 1);

    // Compute the control
    controller.computeControl(x);

    // Save the trajectory from the nominal state
    auto nominal_trajectory = controller.getStateSeq();

    // Save the ancillary trajectory
    auto ancillary_trajectory = controller.getAncillaryStateSeq();


    for (int i = 0; i < num_timesteps; i++) {
      for (int j = 0; j < DYNAMICS::STATE_DIM; j++) {
        actual_trajectory_save[t * num_timesteps * DYNAMICS::STATE_DIM +
                               i*DYNAMICS::STATE_DIM + j] = x(j);
        ancillary_trajectory_save[t * num_timesteps * DYNAMICS::STATE_DIM +
                                  i*DYNAMICS::STATE_DIM + j] = ancillary_trajectory(j, i);
        nominal_trajectory_save[t * num_timesteps * DYNAMICS::STATE_DIM +
                                i*DYNAMICS::STATE_DIM + j] = nominal_trajectory(j, i);
      }
    }
    // Get the open loop control
    DYNAMICS::control_array current_control = controller.getControlSeq().col(0);
//    std::cout << "Current OL control: " << current_control.transpose() << std::endl;


    // Apply the feedback given the current state
    current_control += controller.getFeedbackGains()[0]*(x - controller.getStateSeq().col(0));

    // Propagate the state forward
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    controller.slideControlSequence(1);
  }

  cnpy::npy_save("robust_sc_large_actual.npy",actual_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
  cnpy::npy_save("robust_sc_ancillary.npy",ancillary_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
  cnpy::npy_save("robust_sc_large_nominal.npy",nominal_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
}

TEST(RMPPITest, RobustMPPILargeVarianceRobustCost) {
  using DYNAMICS = DoubleIntegratorDynamics;
  using COST_T = DoubleIntegratorRobustCost;
  // Noise enters the system during the "true" state propagation. In this case the noise is nominal
  DYNAMICS model(100);  // Initialize the double integrator dynamics
  COST_T cost;  // Initialize the cost function
  auto params = cost.getParams();
  params.velocity_desired = 2;
  params.crash_cost = 100;
  cost.setParams(params);
  float dt = 0.02; // Timestep of dynamics propagation
  int max_iter = 3; // Maximum running iterations of optimization
  float lambda = 2.0; // Learning rate parameter
  float alpha = 0.0;
  int crash_status[1] = {0};
  const int num_timesteps = 50;  // Optimization time horizon
  const int total_time_horizon = 5000;

  std::vector<float> actual_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);
  std::vector<float> nominal_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);
  std::vector<float> ancillary_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);
  std::vector<float> feedback_trajectory_save(num_timesteps*total_time_horizon*DoubleIntegratorDynamics::STATE_DIM);

  // Set the initial state
  DYNAMICS::state_array x;
  x << 2, 0, 0, 1;

  DYNAMICS::state_array xdot;

  // control variance
  DYNAMICS::control_array control_var;
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
  Q = 500*DoubleIntegratorDynamics::dfdx::Identity();
  Q(2,2) = 100;
  Q(3,3) = 100;
  /**
   * R = I
   */
  R = 1*DoubleIntegratorCircleCost::control_matrix::Identity();

  /**
   * Qf = I
   */
  Qf = DoubleIntegratorDynamics::dfdx::Identity();

  // Value function threshold
  float value_function_threshold = 10.0;

  // DoubleIntegratorRobustCost cost2;
  // auto controller2 = RobustMPPIController<DYNAMICS, DoubleIntegratorRobustCost, num_timesteps,
  //         1024, 64, 8, 1>(&model, &cost2, dt, max_iter, gamma, value_function_threshold, Q, Qf, R, control_var);

  // Initialize the R MPPI controller
  auto controller = RobustMPPIController<DYNAMICS, COST_T, num_timesteps,
          1024, 64, 8, 1>(&model, &cost, dt, max_iter, lambda, alpha, value_function_threshold, Q, Qf, R, control_var);

  int fail_count = 0;

  // Start the while loop
  for (int t = 0; t < total_time_horizon; ++t) {
    if (cost.computeStateCost(x, t, crash_status) > 1000) {
      fail_count++;
      crash_status[0] = 0;
    }

    if (tubeFailure(x.data())) {
      cnpy::npy_save("robust_rc_large_actual.npy", actual_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
      cnpy::npy_save("robust_rc_ancillary.npy", ancillary_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
      cnpy::npy_save("robust_rc_large_nominal.npy",nominal_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
      cnpy::npy_save("robust_rc_large_feedback.npy",feedback_trajectory_save.data(),
                     {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
      printf("Current Time: %f    ", t * dt);
      model.printState(x.data());
      std::cout << "                          Candidate Free Energies: " << controller.getCandidateFreeEnergy().transpose() << std::endl;
      std::cout << "Tube failure!!" << std::endl;
      FAIL() << "Visualize the trajectories by running scripts/double_integrator/plot_DI_test_trajectories; "
                "the argument to this python file is the build directory of MPPI-Generic";
    }
    // Update the importance sampler
    controller.updateImportanceSamplingControl(x, 1);

    // Compute the control
    controller.computeControl(x, 1);
    controller.computeFeedbackPropagatedStateSeq();

    // Print the system state
    if (t % 100 == 0) {
      printf("Current Time: %f    ", t * dt);
      model.printState(x.data());
      auto free_energy_stats = controller.getFreeEnergyStatistics();
      std::cout << "                           Candidate Free Energies: " << controller.getCandidateFreeEnergy().transpose() << std::endl;
      std::cout << "Real    FE [mean, variance]: [" << free_energy_stats.real_sys.freeEnergyMean << ", " << free_energy_stats.real_sys.freeEnergyVariance << "]" << std::endl;
      std::cout << "Nominal FE [mean, variance]: [" << free_energy_stats.nominal_sys.freeEnergyMean << ", " << free_energy_stats.nominal_sys.freeEnergyVariance << "]" << std::endl;
      std::cout << "Algorithm Health Normalizer: [" << controller.getNormalizerPercent() << "]" << std::endl;
      std::cout << "DF(x, x0, u): [" << controller.computeDF() << "]\n" << std::endl;
    }

    // Save the trajectory from the nominal state
    auto nominal_trajectory = controller.getStateSeq();

    // Save the ancillary trajectory
    auto ancillary_trajectory = controller.getAncillaryStateSeq();

    // Compute the propagated state trajectory
    auto propagated_trajectory = controller.getFeedbackPropagatedStateSeq();

    // Compute the actual trajectory with no feedback
    auto actual_trajectory = controller.getStateSeq();
    controller.computeStateTrajectoryHelper(actual_trajectory, x, controller.getControlSeq());


    for (int i = 0; i < num_timesteps; i++) {
      for (int j = 0; j < DoubleIntegratorDynamics::STATE_DIM; j++) {
        actual_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                               i*DoubleIntegratorDynamics::STATE_DIM + j] = actual_trajectory(j,i);
        ancillary_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                                  i*DoubleIntegratorDynamics::STATE_DIM + j] = ancillary_trajectory(j, i);
        nominal_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                                i*DoubleIntegratorDynamics::STATE_DIM + j] = nominal_trajectory(j, i);
        feedback_trajectory_save[t * num_timesteps * DoubleIntegratorDynamics::STATE_DIM +
                                i*DoubleIntegratorDynamics::STATE_DIM + j] = propagated_trajectory(j, i);
      }
    }
    // Get the open loop control
    DYNAMICS::control_array current_control = controller.getControlSeq().col(0);
//    std::cout << "Current OL control: " << current_control.transpose() << std::endl;


    // Apply the feedback given the current state
    current_control += controller.getFeedbackGains()[0]*(x - controller.getStateSeq().col(0));

    // Compute real free energy bound

    // Nominal free energy bound is always alpha

    // Bound of real free energy growth

    // Propagate the state forward
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);



    // Add the "true" noise of the system
    model.computeStateDisturbance(dt, x);

    // Slide the control sequence
    controller.slideControlSequence(1);
  }

  cnpy::npy_save("robust_rc_large_actual.npy",actual_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
  cnpy::npy_save("robust_rc_ancillary.npy",ancillary_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DYNAMICS::STATE_DIM},"w");
  cnpy::npy_save("robust_rc_large_nominal.npy",nominal_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
  cnpy::npy_save("robust_rc_large_feedback.npy",feedback_trajectory_save.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
}
