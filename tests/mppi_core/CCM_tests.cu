#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/core/rmppi_kernel_test.cuh>
#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <mppi/utils/test_helper.h>

const int NUM_TIMESTEPS = 50;
const int NUM_ROLLOUTS = 64;

// Might be simpler to create a new Controller CLass from RMPPI
template<class DYN_T = DoubleIntegratorDynamics, class COST_T = DoubleIntegratorCircleCost,
         int MAX_TIMESTEPS = NUM_TIMESTEPS, int NUM_ROLLOUTS_T = NUM_ROLLOUTS,
         int B_X = 64, int B_Y = 1, int S  = 1>
class RMPPICCMDoubleIntegratorController : public RobustMPPIController<DYN_T, COST_T,
    MAX_TIMESTEPS, NUM_ROLLOUTS_T, B_X, B_Y, S> {
public:
  using Q_MAT = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                              NUM_ROLLOUTS_T, B_X, B_Y,
                                              S>::StateCostWeight;

  using Qf_MAT = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                               NUM_ROLLOUTS_T, B_X, B_Y,
                                               S>::Hessian;

  using R_MAT = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                              NUM_ROLLOUTS_T, B_X, B_Y,
                                              S>::ControlCostWeight;

  using control_array = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                                      NUM_ROLLOUTS_T, B_X, B_Y,
                                                      S>::control_array;

  using control_trajectory = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                                      NUM_ROLLOUTS_T, B_X, B_Y,
                                                      S>::control_trajectory;

  using state_array = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                                      NUM_ROLLOUTS_T, B_X, B_Y,
                                                      S>::state_array;

  // Constructor... Yeah It ain't pretty
  RMPPICCMDoubleIntegratorController(DYN_T* model, COST_T* cost, float dt, float lambda,
      float alpha, float value_function_threshold,
      const Eigen::Ref<const control_array>& control_std_dev,
      int num_timesteps = MAX_TIMESTEPS,
      const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
      int num_candidate_nominal_states = 9, int optimization_stride = 1,
      cudaStream_t stream = nullptr) : RobustMPPIController<DYN_T, COST_T,
      MAX_TIMESTEPS, NUM_ROLLOUTS_T, 64, 1, 1>(model, cost, dt, 1, lambda, alpha,
      value_function_threshold, Q_MAT::Zero(), Qf_MAT::Zero(), R_MAT::Zero(),
      control_std_dev, num_timesteps, init_control_traj,
      num_candidate_nominal_states, optimization_stride, stream) {
    int help = 0;
  }

  void computeControl(const Eigen::Ref<const state_array>& state) override {
    std::cout << "PPPBBBBBBBBBTTTTTTTTTTTTTT" << std::endl;
    // Rewrite computeControl using the CCM Rollout Kernel
  }
};

TEST(CCMTest, RMPPIRolloutKernel) {
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  DYN model;
  COST cost;
  const int num_timesteps = 50;
  const int num_rollouts = 64;

  using CONTROLLER = RMPPICCMDoubleIntegratorController<DYN, COST,
                                                        num_timesteps,
                                                        num_rollouts>;

  // Todo create RMPPI controller to use updateImportanceSamplingmethods

  const int state_dim = DYN::STATE_DIM;
  const int control_dim = DYN::CONTROL_DIM;

  float dt = 0.01;
  // int max_iter = 10;
  float lambda = 0.1;

  // float x[num_rollouts * state_dim * 2];
  // float x_dot[num_rollouts * state_dim * 2];
  // float u[num_rollouts * control_dim * 2];
  // float du[num_rollouts * control_dim * 2];
  float sigma_u[control_dim] = {0.5, 0.05}; // variance to sample noise from
  CONTROLLER::control_array control_std_dev = CONTROLLER::control_array::Constant(0.5);
  // COST::control_matrix cost_variance = COST::control_matrix::Identity();
  // for(int i = 0; i < control_dim; i++) {
  //   cost_variance(i, i) = sigma_u[i];
  // }
  // float fb_u[num_rollouts * control_dim * state_dim];

  DYN::state_array x_init_act;
  x_init_act << 4, 0, 0, 0;
  DYN::state_array x_init_nom;
  x_init_nom << 0, 0, 0.1, 0;

  // Generate control noise
  float sampled_noise[num_rollouts * num_timesteps * control_dim];
  std::mt19937 rng_gen;
  std::vector<std::normal_distribution<float>> control_dist;
  for (int i = 0; i < control_dim; i++) {
    control_dist.push_back(std::normal_distribution<float>(0, 1));
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
  float u_traj[num_timesteps * control_dim] = {0};
  for (int i = 0; i < num_timesteps; i++) {
    u_traj[i * control_dim] = 1;
  }
  // u_traj[0] = 1;
  // u_traj[1] = 0.5;

  // u_traj[10] = 1;
  // u_traj[11] = 0.5;

  // u_traj[14] = -1;
  // u_traj[15] = 0.5;

  // TODO: Generate feedback gain trajectories
  // VanillaMPPIController<DYN, COST, 100, 512, 64, 8>::feedback_gain_trajectory feedback_gains;
  // for (int i = 0; i < num_timesteps; i++) {
  //   feedback_gains.push_back(DYN::feedback_matrix::Constant(-15));
  // }

  // // Copy Feedback Gains into an array
  // float feedback_array[num_timesteps * control_dim * state_dim];
  // for (size_t i = 0; i < feedback_gains.size(); i++) {
  //   // std::cout << "Matrix " << i << ":\n";
  //   // std::cout << feedback_gains[i] << std::endl;
  //   int i_index = i * control_dim * state_dim;

  //   for (size_t j = 0; j < control_dim * state_dim; j++) {
  //     feedback_array[i_index + j] = feedback_gains[i].data()[j];
  //   }
  // }
  /**
   * Create vectors of data for GPU/CPU test
   */
  std::vector<float> x_init_act_vec, x_init_nom_vec, sigma_u_vec, u_traj_vec;
  x_init_act_vec.assign(x_init_act.data(), x_init_act.data() + state_dim);
  x_init_nom_vec.assign(x_init_nom.data(), x_init_nom.data() + state_dim);
  sigma_u_vec.assign(sigma_u, sigma_u + control_dim);
  u_traj_vec.assign(u_traj, u_traj + num_timesteps * control_dim);
  std::vector<float> feedback_gains_seq_vec, sampled_noise_vec;
  // feedback_gains_seq_vec.assign(feedback_array, feedback_array +
  //   num_timesteps * control_dim * state_dim);
  sampled_noise_vec.assign(sampled_noise, sampled_noise +
    num_rollouts * num_timesteps * control_dim);

  float value_func_threshold = 50000;


  // ============= Entire Sim loop ================
  // for (int i = 0; i < num_timesteps; i++) {
    // UpdateImportanceSamplingControl with new state
    // =============== computeControl ===============
    // Generate noise for new control sequences
    // Call Rollout kernel with new state and noise
    // computeBaseline for nominal and actual

    // Launch NormExpKernel for nominal and actual
    // Compute Normalizer for norminal and actual
    // Launch weighted reduction kernel for nominal and actual
    // Optional: Smooth trajectory

    // ============== Update State ======================
    // Get current control
    // Get current feedback
    // ComputeDynamics
    // Update State

  // }
  // Output Trajectory Costs
  std::array<float, num_rollouts> costs_act_CPU, costs_nom_CPU;
  launchRMPPIRolloutKernelCCMCPU<DYN, COST, num_rollouts>(&model, &cost, dt,
    num_timesteps, lambda, value_func_threshold, x_init_nom_vec, x_init_act_vec,
    sigma_u_vec, u_traj_vec, sampled_noise_vec,
    costs_act_CPU, costs_nom_CPU);
}