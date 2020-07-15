#include <gtest/gtest.h>
#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <mppi/core/rmppi_kernel_test.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/feedback_controllers/CCM/ccm.h>
#include <mppi/utils/test_helper.h>

const int NUM_TIMESTEPS = 50;
const int NUM_ROLLOUTS_CONST = 1024;

// Might be simpler to create a new Controller CLass from RMPPI
template<class DYN_T = DoubleIntegratorDynamics, class COST_T = DoubleIntegratorCircleCost,
         int MAX_TIMESTEPS = NUM_TIMESTEPS, int NUM_ROLLOUTS = NUM_ROLLOUTS_CONST,
         int B_X = 64, int B_Y = 1, int S = 1>
class RMPPICCMDoubleIntegratorController : public RobustMPPIController<DYN_T, COST_T,
    MAX_TIMESTEPS, NUM_ROLLOUTS, B_X, B_Y, S> {
public:
  using Q_MAT = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                              NUM_ROLLOUTS, B_X, B_Y,
                                              S>::StateCostWeight;

  using Qf_MAT = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                               NUM_ROLLOUTS, B_X, B_Y,
                                               S>::Hessian;

  using R_MAT = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                              NUM_ROLLOUTS, B_X, B_Y,
                                              S>::ControlCostWeight;

  using control_array = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                                      NUM_ROLLOUTS, B_X, B_Y,
                                                      S>::control_array;

  using control_trajectory = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                                      NUM_ROLLOUTS, B_X, B_Y,
                                                      S>::control_trajectory;

  using state_array = typename RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS,
                                                      NUM_ROLLOUTS, B_X, B_Y,
                                                      S>::state_array;

  // Constructor... Yeah It ain't pretty
  RMPPICCMDoubleIntegratorController(DYN_T* model, COST_T* cost, float dt, float lambda,
      float alpha, float value_function_threshold,
      const Eigen::Ref<const control_array>& control_std_dev,
      int num_timesteps = MAX_TIMESTEPS,
      const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
      int num_candidate_nominal_states = 9, int optimization_stride = 1,
      cudaStream_t stream = nullptr) : RobustMPPIController<DYN_T, COST_T,
      MAX_TIMESTEPS, NUM_ROLLOUTS, 64, 1, 1>(model, cost, dt, 1, lambda, alpha,
      value_function_threshold, Q_MAT::Zero(), Qf_MAT::Zero(), R_MAT::Zero(),
      control_std_dev, num_timesteps, init_control_traj,
      num_candidate_nominal_states, optimization_stride, stream) {

    control_dist_ = std::normal_distribution<float>(0, 1);
    CCM_feedback_controller_ = std::make_shared<ccm::LinearCCM<DYN_T>>(model);
  }

  void ptrToVec(const float* input, int num, std::vector<float>& output) {
    std::cout << "ptrToVec before Size: " << output.size() << std::endl;
    output.assign(input, input + num);
    if (output.size() != num) {
      std::cout << "Hit weird case" << std::endl;
      output.assign(num, 0.0);
      for(int i = 0; i < num; i++) {
        output[i] = input[i];
      }
    }
    std::cout << "ptrToVec after Size: " << output.size() << std::endl;

  }

  void computeControl(const Eigen::Ref<const state_array>& state) override {
    std::cout << "PPPBBBBBBBBBTTTTTTTTTTTTTT" << std::endl;
    // Rewrite computeControl using the CCM Rollout Kernel
    int c_dim = DYN_T::CONTROL_DIM;
    int s_dim = DYN_T::STATE_DIM;
    int single_control_traj_size = this->num_timesteps_ * c_dim;
    int multi_control_traj_size = NUM_ROLLOUTS * single_control_traj_size;

    // Handy dandy pointers to nominal data
    float * trajectory_costs_nominal_d = this->trajectory_costs_d_ + NUM_ROLLOUTS;
    float * initial_state_nominal_d = this->initial_state_d_ + s_dim;
    float * control_noise_nominal_d = this->control_noise_d_ + multi_control_traj_size;
    float * control_nominal_d = this->control_d_ + single_control_traj_size;
    for (int opt_iter = 0; opt_iter < this->num_iters_; opt_iter++) {
      // Create noise for trajectories
      std::vector<float> control_noise_vec(multi_control_traj_size * 2, 0);
      for(int i = 0; i < multi_control_traj_size; i++) {
        control_noise_vec[i] = control_dist_(rng_gen_);
        control_noise_vec[multi_control_traj_size + i] = control_noise_vec[i];
      }
      std::vector<float> x_init_act_vec, x_init_nom_vec, u_traj_vec;
      std::vector<float> control_std_dev_vec;
      std::cout << "State: " << state.transpose() << std::endl;
      ptrToVec(state.data(), s_dim, x_init_act_vec);
      ptrToVec(this->nominal_state_.data(), s_dim, x_init_nom_vec);
      std::cout << "NOMINAL: " << x_init_nom_vec.size() << std::endl;
      ptrToVec(this->nominal_control_trajectory_.data(),
               single_control_traj_size, u_traj_vec);
      ptrToVec(this->control_std_dev_.data(), c_dim, control_std_dev_vec);


      // Launch rollout kernel using CCM
      // TODO pass in alpha
      std::array<float, NUM_ROLLOUTS> costs_act_CPU, costs_nom_CPU;
      std::cout << "To my childhood philistines" << std::endl;
      launchRMPPIRolloutKernelCCMCPU<DYN_T, COST_T, NUM_ROLLOUTS>(this->model_,
        this->cost_, CCM_feedback_controller_.get(), this->dt_,
        this->num_timesteps_, this->lambda_, this->alpha_,
        this->value_function_threshold_, x_init_nom_vec, x_init_act_vec,
        control_std_dev_vec, u_traj_vec, control_noise_vec,
        costs_act_CPU, costs_nom_CPU);

      std::cout << "Oh my god I thought you would never ask" << std::endl;

      for(int i = 0; i < multi_control_traj_size; i++) {
        this->trajectory_costs_(i) = costs_act_CPU[i];
        this->trajectory_costs_nominal_(i) = costs_nom_CPU[i];
      }
      // Control noise should be modified to contain u + noise
      this->baseline_ = mppi_common::computeBaselineCost(
          this->trajectory_costs_.data(), NUM_ROLLOUTS);
      this->baseline_nominal_ = mppi_common::computeBaselineCost(
          this->trajectory_costs_nominal_.data(), NUM_ROLLOUTS);

      // Copy data over to GPU
      HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_d_,
                                  this->trajectory_costs_.data(),
                                  NUM_ROLLOUTS * sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_));

      HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_d,
                                  this->trajectory_costs_nominal_.data(),
                                  NUM_ROLLOUTS * sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_));

      HANDLE_ERROR(cudaMemcpyAsync(this->control_noise_d_,
                                  control_noise_vec.data(),
                                  multi_control_traj_size * sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_));

      HANDLE_ERROR(cudaMemcpyAsync(control_noise_nominal_d,
                                  control_noise_vec.data() + multi_control_traj_size,
                                  multi_control_traj_size * sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_));

      // After rollout kernel, control_d_ and nominal_control_d are written to
      // and not read from so there is nothing to copy to them
      // HANDLE_ERROR(cudaMemcpyAsync(this->control_d_,
      //                              this->nominal_control_trajectory_.data(),
      //                              single_control_traj_size * sizeof(float),
      //                              cudaMemcpyHostToDevice, this->stream_));

      // // TODO Not done in RMPPI RolloutKernel but I think it should be
      // HANDLE_ERROR(cudaMemcpyAsync(control_nominal_d,
      //                              this->nominal_control_trajectory_.data(),
      //                              single_control_traj_size * sizeof(float),
      //                              cudaMemcpyHostToDevice, this->stream_));

      HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

      std::cout << "What's the occasion?" << std::endl;

      // In this case this->gamma = 1 / lambda
      mppi_common::launchNormExpKernel(NUM_ROLLOUTS, B_X,
                                      this->trajectory_costs_d_, this->lambda_,
                                      this->baseline_, this->stream_);
      mppi_common::launchNormExpKernel(NUM_ROLLOUTS, B_X,
                                      trajectory_costs_nominal_d, this->lambda_,
                                      this->baseline_nominal_, this->stream_);

      HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(),
                                  this->trajectory_costs_d_,
                                  NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost,
                                  this->stream_));
      HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_nominal_.data(),
                                  trajectory_costs_nominal_d,
                                  NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost,
                                  this->stream_));
      HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

      std::cout << "What's a little scandal between friends?" << std::endl;

      this->normalizer_ = mppi_common::computeNormalizer(
          this->trajectory_costs_.data(), NUM_ROLLOUTS);
      this->normalizer_nominal_ = mppi_common::computeNormalizer(
          this->trajectory_costs_nominal_.data(), NUM_ROLLOUTS);


      mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, B_X>(
              this->trajectory_costs_d_, this->control_noise_d_, this->control_d_,
              this->normalizer_, this->num_timesteps_, this->stream_);
      mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, B_X>(
              trajectory_costs_nominal_d,
              control_noise_nominal_d, control_nominal_d,
              this->normalizer_nominal_, this->num_timesteps_, this->stream_);
      std::cout << "I don't know you" << std::endl;
      // Transfer the new control to the host
      HANDLE_ERROR( cudaMemcpyAsync(this->control_.data(), this->control_d_,
                                    sizeof(float) * single_control_traj_size,
                                    cudaMemcpyDeviceToHost, this->stream_));
      HANDLE_ERROR( cudaMemcpyAsync(this->nominal_control_trajectory_.data(),
                                    control_nominal_d,
                                    sizeof(float) * single_control_traj_size,
                                    cudaMemcpyDeviceToHost, this->stream_));
      cudaStreamSynchronize(this->stream_);
    }
    std::cout << "Finished Computing control" << std::endl;
  }

  control_array getCCMFeedbackGains(const Eigen::Ref<const state_array>& x_act,
                                 const Eigen::Ref<const state_array>& x_nom,
                                 const Eigen::Ref<const control_array>& u_nom) {
    std::cout << "M:\n" << CCM_feedback_controller_->M(state_array::Zero()) << std::endl;
    control_array fb_u = CCM_feedback_controller_->u_feedback(x_act, x_nom, u_nom);
    return fb_u;

  }
protected:
  std::mt19937 rng_gen_;
  std::normal_distribution<float> control_dist_;
  std::shared_ptr<ccm::LinearCCM<DYN_T>> CCM_feedback_controller_;
};

TEST(CCMTest, RMPPIRolloutKernel) {
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  DYN model;
  COST cost;
  const int num_timesteps = NUM_TIMESTEPS;
  const int num_rollouts = NUM_ROLLOUTS_CONST;

  using CONTROLLER = RMPPICCMDoubleIntegratorController<DYN, COST,
                                                        num_timesteps,
                                                        num_rollouts>;

  // Todo create RMPPI controller to use updateImportanceSamplingmethods

  const int state_dim = DYN::STATE_DIM;
  const int control_dim = DYN::CONTROL_DIM;

  float dt = 0.01;
  // int max_iter = 10;
  float lambda = 0.1;
  float alpha = 0;
  float value_func_threshold = 50000;

  CONTROLLER::control_array control_std_dev = CONTROLLER::control_array::Constant(0.5);
  CONTROLLER::control_trajectory u_traj_eigen = CONTROLLER::control_trajectory::Zero();
  // Set first control to 1 across entire time
  u_traj_eigen.row(0) = CONTROLLER::cost_trajectory::Constant(1.0);

  CONTROLLER rmppi_controller = CONTROLLER(&model, &cost, dt, lambda, alpha,
                                           value_func_threshold,
                                           control_std_dev, num_timesteps,
                                           u_traj_eigen);


  // float x[num_rollouts * state_dim * 2];
  // float x_dot[num_rollouts * state_dim * 2];
  // float u[num_rollouts * control_dim * 2];
  // float du[num_rollouts * control_dim * 2];
  // float sigma_u[control_dim] = {0.5, 0.05}; // variance to sample noise from

  // COST::control_matrix cost_variance = COST::control_matrix::Identity();
  // for(int i = 0; i < control_dim; i++) {
  //   cost_variance(i, i) = sigma_u[i];
  // }
  // float fb_u[num_rollouts * control_dim * state_dim];

  DYN::state_array x_init_act, x_dot;
  x_init_act << 4, 0, 0, 0;
  DYN::state_array x_init_nom;
  x_init_nom << 0, 0, 0.1, 0;

  int mission_length = int(10 / dt); // 10 seconds

  // rmppi_controller.computeControl(x_init_act);
  DYN::state_array x = x_init_act;

  for(int t = 0; t < mission_length; t++) {
    if (t % 100 == 0) {
      printf("Current Time: %f    ", t * dt);
      model.printState(x.data());
      std::cout << "                          Candidate Free Energies: "
                << rmppi_controller.getCandidateFreeEnergy().transpose()
                << std::endl;
    }
    printf("I just want you to watch me dissolve\n");
    rmppi_controller.updateImportanceSamplingControl(x, 1);
    printf("Can't find an exit\n");
    rmppi_controller.computeControl(x);
    printf("So baby come on home\n");
    DYN::state_array x_nom = rmppi_controller.getStateSeq().col(0);
    DYN::control_array current_control = rmppi_controller.getControlSeq().col(0);
    printf("Where the tick tock of my heart is beating right on time\n");
    current_control += rmppi_controller.getCCMFeedbackGains(x, x_nom, current_control);
    printf("I leave a light outside\n");
    model.computeDynamics(x, current_control, x_dot);
    printf("And I hope that you know that you don't have to hide\n");
    model.updateState(x, x_dot, dt);

    model.computeStateDisturbance(dt, x);

    rmppi_controller.slideControlSequence(1);


  }


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
  sigma_u_vec.assign(control_std_dev.data(), control_std_dev.data() + control_dim);
  u_traj_vec.assign(u_traj, u_traj + num_timesteps * control_dim);
  std::vector<float> feedback_gains_seq_vec, sampled_noise_vec;
  // feedback_gains_seq_vec.assign(feedback_array, feedback_array +
  //   num_timesteps * control_dim * state_dim);
  sampled_noise_vec.assign(sampled_noise, sampled_noise +
    num_rollouts * num_timesteps * control_dim);


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
  // std::array<float, num_rollouts> costs_act_CPU, costs_nom_CPU;
  // launchRMPPIRolloutKernelCCMCPU<DYN, COST, num_rollouts>(&model, &cost, dt,
  //   num_timesteps, lambda, alpha, value_func_threshold, x_init_nom_vec,
  //   x_init_act_vec, sigma_u_vec, u_traj_vec, sampled_noise_vec,
  //   costs_act_CPU, costs_nom_CPU);
}
