#include <gtest/gtest.h>
#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <mppi/core/rmppi_kernel_test.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/feedback_controllers/CCM/ccm.h>
#include <mppi/utils/test_helper.h>
#include <cnpy.h>

const int NUM_TIMESTEPS = 50;
const int NUM_ROLLOUTS_CONST = 1024;

// Might be simpler to create a new Controller CLass from RMPPI
template<class DYN_T = DoubleIntegratorDynamics, class COST_T = DoubleIntegratorCircleCost,
         int MAX_TIMESTEPS = NUM_TIMESTEPS, int NUM_ROLLOUTS = NUM_ROLLOUTS_CONST,
         int B_X = 64, int B_Y = 1, int S = 1>
class RMPPICCMDoubleIntegratorController : public RobustMPPIController<DYN_T, COST_T,
    MAX_TIMESTEPS, NUM_ROLLOUTS, B_X, B_Y, S> {

protected:
  std::mt19937 rng_gen_;
  std::normal_distribution<float> control_dist_;
  ccm::LinearCCM<DYN_T> CCM_feedback_controller_;
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
    CCM_feedback_controller_ = ccm::LinearCCM<DYN_T>(model);
    // Q_MAT M_new = Q_MAT::Identity();
    // CCM_feedback_controller_.setM(M_new);
  }

  void ptrToVec(const float* input, int num, std::vector<float>& output) {
    output.assign(input, input + num);
    if (output.size() != num) {
      output.assign(num, 0.0);
      for(int i = 0; i < num; i++) {
        output[i] = input[i];
      }
    }

  }

  void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride=1) override {
    std::cout << "Candidate chosen: " << this->best_index_ << std::endl;
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
      ptrToVec(state.data(), s_dim, x_init_act_vec);
      ptrToVec(this->nominal_state_.data(), s_dim, x_init_nom_vec);
      ptrToVec(this->nominal_control_trajectory_.data(),
               single_control_traj_size, u_traj_vec);
      ptrToVec(this->control_std_dev_.data(), c_dim, control_std_dev_vec);


      // Launch rollout kernel using CCM
      // TODO pass in alpha
      std::array<float, NUM_ROLLOUTS> costs_act_CPU, costs_nom_CPU;
      launchRMPPIRolloutKernelCCMCPU<DYN_T, COST_T, NUM_ROLLOUTS>(this->model_,
        this->cost_, &CCM_feedback_controller_, this->dt_,
        this->num_timesteps_, optimization_stride, this->lambda_, this->alpha_,
        this->value_function_threshold_, x_init_nom_vec, x_init_act_vec,
        control_std_dev_vec, u_traj_vec, control_noise_vec,
        costs_act_CPU, costs_nom_CPU);

      for(int i = 0; i < NUM_ROLLOUTS; i++) {
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

      // In this case this->gamma = 1 / lambda
      mppi_common::launchNormExpKernel(NUM_ROLLOUTS, B_X,
                                      this->trajectory_costs_d_, 1.0 / this->lambda_,
                                      this->baseline_, this->stream_);
      mppi_common::launchNormExpKernel(NUM_ROLLOUTS, B_X,
                                      trajectory_costs_nominal_d, 1.0 / this->lambda_,
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
  }

  void computeNominalFeedbackGains(const Eigen::Ref<const state_array>& state) override {}

  control_array getCCMFeedbackGains(const Eigen::Ref<const state_array>& x_act,
                                 const Eigen::Ref<const state_array>& x_nom,
                                 const Eigen::Ref<const control_array>& u_nom) {
    control_array fb_u = CCM_feedback_controller_.u_feedback(x_act, x_nom, u_nom);
    std::cout << "Act: " << x_act.transpose() << std::endl;
    std::cout << "nom: " << x_nom.transpose() << std::endl;
    std::cout << "U: " << u_nom.transpose() << std::endl;
    std::cout << "Feedback: " << fb_u.transpose() << std::endl;
    return fb_u;

  }
};

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

TEST(CCMTest, CCMFeedbackTest) {
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  DYN model(100);
  ccm::LinearCCM<DYN> fb_controller(&model);
  float dt = 0.02;
  int mission_length = int(10/dt);

  DYN::state_array x, x_nom, x_dot;
  x << 4, 0, 0, 1;
  x_nom << -3, 2, 0, 0;
  DYN::control_array current_control;

  float two_percent_settle_time = -1;

  for (int t = 0; t < mission_length; t++) {
    current_control = fb_controller.u_feedback(x, x_nom, DYN::control_array::Zero());
    model.computeDynamics(x, current_control, x_dot);
    model.updateState(x, x_dot, dt);

    DYN::state_array abs_diff = x - x_nom;
    for (int i = 0; i < DYN::STATE_DIM; i++) {
      if (x_nom(i) >= 1) {
        abs_diff(i) /= x_nom(i);
      }
    }
    abs_diff = abs_diff.cwiseAbs();
    if (abs_diff.block<2,1>(0,0).maxCoeff() < 0.02 && two_percent_settle_time < 0) {
      two_percent_settle_time = t * dt;
    }

    if (t % 5 == 0) {
      std::cout << "State at t = " << t * dt << ": " << x.transpose() << std::endl;
    }
  }
  std::cout << "2% settling time is " << two_percent_settle_time << " secs" << std::endl;
}

TEST(CCMTest, RMPPIRolloutKernel) {
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  DYN model(100);
  COST cost;
  const int num_timesteps = NUM_TIMESTEPS;
  const int num_rollouts = NUM_ROLLOUTS_CONST;

  using CONTROLLER = RMPPICCMDoubleIntegratorController<DYN, COST,
                                                        num_timesteps,
                                                        num_rollouts>;

  const int state_dim = DYN::STATE_DIM;
  const int control_dim = DYN::CONTROL_DIM;

  float dt = 0.02;
  // int max_iter = 10;
  float lambda = 4;
  float alpha = 0;
  float value_func_threshold = 10;


  int mission_length = int(100 / dt); // 100 seconds

  std::vector<float> actual_trajectory_save(num_timesteps*mission_length*DYN::STATE_DIM);
  std::vector<float> nominal_trajectory_save(num_timesteps*mission_length*DYN::STATE_DIM);
  std::string file_prefix = "/data/bvlahov3/RMPPI_CCM_control_trajectories_CoRL2020/";

  CONTROLLER::control_array control_std_dev = CONTROLLER::control_array::Constant(1.0);
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
  x_init_act << 2, 0, 0, 1;
  DYN::state_array x_init_nom;
  x_init_nom << 0, 0, 0.1, 0;

  // rmppi_controller.computeControl(x_init_act);
  DYN::state_array x = x_init_act;

  for(int t = 0; t < mission_length; t++) {
    std::string act_file_name = file_prefix + "robust_large_actual_CCM_t_" +
                                std::to_string(t) + ".npy";
    std::string nom_file_name = file_prefix + "robust_large_nominal_CCM_t_" +
                                std::to_string(t) + ".npy";

    // if (cost.computeStateCost(x) > 1000) {
    //   std::cout << "State Cost is " << cost.computeStateCost(x) << std::endl;
    //   std::cout << "State was " << x.transpose() << std::endl;
    //   FAIL();
    // }

    if (tubeFailure(x.data())) {
      cnpy::npy_save(act_file_name, actual_trajectory_save.data(),
                     {mission_length, num_timesteps, DYN::STATE_DIM},"w");
      cnpy::npy_save(nom_file_name, nominal_trajectory_save.data(),
                     {mission_length, num_timesteps, DYN::STATE_DIM},"w");
      printf("Current Time: %f    ", t * dt);
      model.printState(x.data());
      std::cout << "\tCandidate Free Energies: "
                << rmppi_controller.getCandidateFreeEnergy().transpose() << std::endl;
      std::cout << "Tube failure!!" << std::endl;
      FAIL() << "Visualize the trajectories by running scripts/double_integrator/plot_DI_test_trajectories; "
                "the argument to this python file is the build directory of MPPI-Generic";
    }

    if (t % 2 == 0) {
      printf("Current Time: %5.2f    ", t * dt);
      model.printState(x.data());
      std::cout << "\tCandidate Free Energies: "
                << rmppi_controller.getCandidateFreeEnergy().transpose()
                << std::endl;
    }
    rmppi_controller.updateImportanceSamplingControl(x, 1);
    rmppi_controller.computeControl(x);

    auto nominal_trajectory = rmppi_controller.getStateSeq();

    for (int i = 0; i < num_timesteps; i++) {
      for (int j = 0; j < DYN::STATE_DIM; j++) {
        actual_trajectory_save[t * num_timesteps * DYN::STATE_DIM +
                               i*DYN::STATE_DIM + j] = x(j);
        nominal_trajectory_save[t * num_timesteps * DYN::STATE_DIM +
                                i*DYN::STATE_DIM + j] = nominal_trajectory(j, i);
      }
    }
    DYN::state_array x_nom = rmppi_controller.getStateSeq().col(0);
    DYN::control_array current_control = rmppi_controller.getControlSeq().col(0);

    current_control += rmppi_controller.getCCMFeedbackGains(x, x_nom, current_control);
    model.computeDynamics(x, current_control, x_dot);
    model.updateState(x, x_dot, dt);

    model.computeStateDisturbance(dt, x);
    rmppi_controller.slideControlSequence(1);

    if (t % 50 == 0) {
      cnpy::npy_save(act_file_name, actual_trajectory_save.data(),
                     {mission_length, num_timesteps, DYN::STATE_DIM},"w");
      cnpy::npy_save(nom_file_name, nominal_trajectory_save.data(),
                     {mission_length, num_timesteps, DYN::STATE_DIM},"w");
    }
  }
}
