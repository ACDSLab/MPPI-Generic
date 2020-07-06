#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/core/rmppi_kernel_test.cuh>
#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <mppi/utils/test_helper.h>

TEST(CCMTest, RMPPIRolloutKernel) {
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  DYN model;
  COST cost;

  const int state_dim = DYN::STATE_DIM;
  const int control_dim = DYN::CONTROL_DIM;

  float dt = 0.01;
  // int max_iter = 10;
  float lambda = 0.1;
  const int num_timesteps = 50;
  const int num_rollouts = 64;

  // float x[num_rollouts * state_dim * 2];
  // float x_dot[num_rollouts * state_dim * 2];
  // float u[num_rollouts * control_dim * 2];
  // float du[num_rollouts * control_dim * 2];
  float sigma_u[control_dim] = {0.5, 0.05}; // variance to sample noise from
  COST::control_matrix cost_variance = COST::control_matrix::Identity();
  for(int i = 0; i < control_dim; i++) {
    cost_variance(i, i) = sigma_u[i];
  }
  // float fb_u[num_rollouts * control_dim * state_dim];

  DYN::state_array x_init_act;
  x_init_act << 0, 0, 0, 0;
  DYN::state_array x_init_nom;
  x_init_nom << 4, 0, 0.1, 0;

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

  // Output Trajectory Costs
  std::array<float, num_rollouts> costs_act_CPU, costs_nom_CPU;
  launchRMPPIRolloutKernelCCMCPU<DYN, COST, num_rollouts>(&model, &cost, dt,
    num_timesteps, lambda, value_func_threshold, x_init_nom_vec, x_init_act_vec,
    sigma_u_vec, u_traj_vec, sampled_noise_vec,
    costs_act_CPU, costs_nom_CPU);
}