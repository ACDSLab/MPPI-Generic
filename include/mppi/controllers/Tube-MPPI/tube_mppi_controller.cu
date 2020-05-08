#include "tube_mppi_controller.cuh"


#define TubeMPPI TubeMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y>
TubeMPPI::TubeMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter, float gamma,
                             const Eigen::Ref<const StateCostWeight>& Q,
                             const Eigen::Ref<const Hessian>& Qf,
                             const Eigen::Ref<const ControlCostWeight>& R,
                             const Eigen::Ref<const control_array>& control_variance,
                             int num_timesteps,
                             const Eigen::Ref<const control_trajectory>& init_control_traj,
                             cudaStream_t stream) :Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
                                     model, cost, dt, max_iter, gamma,
                                     control_variance, num_timesteps, init_control_traj, stream) {

  nominal_control_trajectory_ = init_control_traj;
  // TODO copy to GPU

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise variance to the device
  this->copyControlVarianceToDevice();

  initDDP(Q, Qf, R);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::computeControl(const Eigen::Ref<const state_array>& state) {
  // TODO
  if (!nominalStateInit_){
//    for (int i = 0; i < DYN_T::STATE_DIM; i++){
//      nominal_state_trajectory_(i, 0) = state(i);
//    }
    nominal_state_trajectory_.col(0) = state;
    nominalStateInit_ = true;
  }

//  std::cout << "Post disturbance Actual State: "; this->model_->printState(state.data());
//  std::cout << "                Nominal State: "; this->model_->printState(nominal_state_trajectory_.col(0).data());

  // Handy reference pointers
  float * trajectory_costs_nominal_d = this->trajectory_costs_d_ + NUM_ROLLOUTS;
  float * initial_state_nominal_d = this->initial_state_d_ + DYN_T::STATE_DIM;

  float * control_noise_nominal_d = this->control_noise_d_ + NUM_ROLLOUTS *
                                    this->num_timesteps_ * DYN_T::CONTROL_DIM;
  float * control_nominal_d = this->control_d_ + this->num_timesteps_ * DYN_T::CONTROL_DIM;

  for (int opt_iter = 0; opt_iter < this->num_iters_; opt_iter++) {
    // Send the initial condition to the device

    HANDLE_ERROR( cudaMemcpyAsync(this->initial_state_d_, state.data(),
                                  DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, this->stream_));

    HANDLE_ERROR( cudaMemcpyAsync(initial_state_nominal_d, nominal_state_trajectory_.data(),
                                  DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, this->stream_));

    // Send the nominal control to the device
    copyControlToDevice();

    //Generate noise data
    curandGenerateNormal(this->gen_, this->control_noise_d_,
                         NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM,
                         0.0, 1.0);

    cudaDeviceSynchronize();

    curandGenerateNormal(this->gen_, control_noise_nominal_d,
                         NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM,
                         0.0, 1.0);
//    HANDLE_ERROR( cudaMemcpyAsync(control_noise_nominal_d, control_noise_d_,
//                 NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM * sizeof(float),
//                 cudaMemcpyDeviceToDevice,
//                 stream_) );

    //Launch the rollout kernel TODO fix the rollout kernel
//    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y, 2>(
//        this->model_->model_d_, this->cost_->cost_d_, dt_, this->num_timesteps_,
//        initial_state_d_, control_d_, control_noise_d_,
//        this->control_variance_d_, trajectory_costs_d_, stream_);

    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
            this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_,
            this->initial_state_d_, this->control_d_, this->control_noise_d_,
            this->control_variance_d_, this->trajectory_costs_d_, this->stream_);

    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
            this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_,
            initial_state_nominal_d, control_nominal_d, control_noise_nominal_d,
            this->control_variance_d_, trajectory_costs_nominal_d, this->stream_);

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(),
        this->trajectory_costs_d_,
        NUM_ROLLOUTS*sizeof(float),
        cudaMemcpyDeviceToHost, this->stream_));

    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(),
        trajectory_costs_nominal_d,
        NUM_ROLLOUTS*sizeof(float),
        cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR( cudaStreamSynchronize(this->stream_) );

    this->baseline_ = mppi_common::computeBaselineCost(
        this->trajectory_costs_.data(),
        NUM_ROLLOUTS);

    baseline_nominal_ = mppi_common::computeBaselineCost(
        this->trajectory_costs_nominal_.data(),
        NUM_ROLLOUTS);

    // Launch the norm exponential kernel for both actual and nominal
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
        this->trajectory_costs_d_, this->gamma_, this->baseline_, this->stream_);

    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
        trajectory_costs_nominal_d, this->gamma_, this->baseline_nominal_, this->stream_);

    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(),
        this->trajectory_costs_d_,
        NUM_ROLLOUTS*sizeof(float),
        cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(),
        trajectory_costs_nominal_d,
        NUM_ROLLOUTS*sizeof(float),
        cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->normalizer_ = mppi_common::computeNormalizer(
        this->trajectory_costs_.data(), NUM_ROLLOUTS);
    normalizer_nominal_ = mppi_common::computeNormalizer(
        this->trajectory_costs_nominal_.data(), NUM_ROLLOUTS);

    // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
    mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
        this->trajectory_costs_d_, this->control_noise_d_, this->control_d_,
        this->normalizer_, this->num_timesteps_, this->stream_);
    mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
        trajectory_costs_nominal_d,
        control_noise_nominal_d, control_nominal_d,
        this->normalizer_nominal_, this->num_timesteps_, this->stream_);

    // Transfer the new control to the host
    HANDLE_ERROR( cudaMemcpyAsync(this->control_.data(), this->control_d_,
            sizeof(float)*this->num_timesteps_*DYN_T::CONTROL_DIM,
                                  cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR( cudaMemcpyAsync(nominal_control_trajectory_.data(), control_nominal_d,
            sizeof(float)*this->num_timesteps_*DYN_T::CONTROL_DIM,
                                  cudaMemcpyDeviceToHost, this->stream_));
    cudaStreamSynchronize(this->stream_);

    // Compute the nominal and actual state trajectories

    computeStateTrajectory(state); // Input is the actual state


    if (this->baseline_ < baseline_nominal_ + nominal_threshold_) {
      // In this case, the disturbance the made the nominal and actual states differ improved the cost.
      // std::copy(state_trajectory.begin(), state_trajectory.end(), nominal_state_trajectory_.begin());
      // std::copy(control_trajectory.begin(), control_trajectory.end(), nominal_control_.begin());
      nominal_state_trajectory_ = this->state_;
      nominal_control_trajectory_ = this->control_;
    }

    // Outside of this loop, we will utilize the nominal state trajectory and the nominal control trajectory to compute
    // the optimal feedback gains using our ancillary controller, then apply feedback inside our main while loop at the
    // same rate as our state estimator.

  }
  smoothControlTrajectory();
  computeStateTrajectory(state); // Input is the actual state
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::copyControlToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(this->control_d_, this->control_.data(),
                                 sizeof(float) * this->control_.size(),
                                 cudaMemcpyHostToDevice, this->stream_));

    HANDLE_ERROR(cudaMemcpyAsync(this->control_d_ + this->control_.size(),
                                 nominal_control_trajectory_.data(),
                                 sizeof(float) * nominal_control_trajectory_.size(),
                                 cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::allocateCUDAMemory() {
  Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::allocateCUDAMemoryHelper(1);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y>
void TubeMPPI::slideControlSequence(int steps) {

  // Save the control history
  if (steps > 1) {
    this->control_history_.row(0) = nominal_control_trajectory_.col(steps - 2).transpose();
    this->control_history_.row(1) = nominal_control_trajectory_.col(steps - 1).transpose();
  } else { //
    this->control_history_.row(0) = this->control_history_.row(1); // Slide control history forward
    this->control_history_.row(1) = nominal_control_trajectory_.col(0).transpose(); // Save the control at time 0
  }

  for (int i = 0; i < this->num_timesteps_; ++i) {
    for (int j = 0; j < DYN_T::CONTROL_DIM; j++) {
      int ind = std::min(i + steps, this->num_timesteps_ - 1);
      nominal_control_trajectory_(j,i) = nominal_control_trajectory_(j, ind);
      this->control_(j,i) = this->control_(j, ind);
    }
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void TubeMPPI::smoothControlTrajectory() {
  // Create the filter coefficients
  Eigen::Matrix<float, 1, 5> filter_coefficients;
  filter_coefficients << -3, 12, 17, 12, -3;
  filter_coefficients /= 35.0;

  // Create and fill a control buffer that we can apply the convolution filter
  Eigen::Matrix<float, MAX_TIMESTEPS+4, DYN_T::CONTROL_DIM> control_buffer;

  // Fill the first two timesteps with the control history
  control_buffer.topRows(2) = this->control_history_;

  // Fill the center timesteps with the current nominal trajectory
  control_buffer.middleRows(2, MAX_TIMESTEPS) = nominal_control_trajectory_.transpose();

  // Fill the last two timesteps with the end of the current nominal control trajectory
  control_buffer.row(MAX_TIMESTEPS+2) = nominal_control_trajectory_.transpose().row(MAX_TIMESTEPS-1);
  control_buffer.row(MAX_TIMESTEPS+3) = control_buffer.row(MAX_TIMESTEPS+2);

  // Apply convolutional filter to each timestep
  for (int i = 0; i < MAX_TIMESTEPS; ++i) {
    nominal_control_trajectory_.col(i) = (filter_coefficients*control_buffer.middleRows(i,5)).transpose();
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::initDDP(const StateCostWeight& q_mat,
                       const Hessian& q_f_mat,
                       const ControlCostWeight& r_mat) {
    util::DefaultLogger logger;
    bool verbose = false;
    ddp_model_  = std::make_shared<ModelWrapperDDP<DYN_T>>(this->model_);
    ddp_solver_ = std::make_shared< DDP<ModelWrapperDDP<DYN_T>>>(this->dt_,
            this->num_timesteps_, 1, &logger, verbose);
    Q_ = q_mat;
    Qf_ = q_f_mat;
    R_ = r_mat;

    for (int i = 0; i < DYN_T::CONTROL_DIM; i++) {
        control_min_(i) = this->model_->control_rngs_[i].x;
        control_max_(i) = this->model_->control_rngs_[i].y;
    }

    run_cost_ = std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(Q_,
        R_, this->num_timesteps_);
    terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(Qf_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::computeFeedbackGains(const Eigen::Ref<const state_array>& state) {

  run_cost_->setTargets(nominal_state_trajectory_.data(), nominal_control_trajectory_.data(),
                        this->num_timesteps_);
//  // Convert state_array to eigen
//  Eigen::Matrix<float, DYN_T::STATE_DIM, 1> s;
//  for (int i = 0; i < DYN_T::STATE_DIM; i++) {
//    s(i) = state[i];
//  }
  terminal_cost_->xf = run_cost_->traj_target_x_.col(this->num_timesteps_ - 1);
  result_ = ddp_solver_->run(state, this->control_,
                             *ddp_model_, *run_cost_, *terminal_cost_,
                             control_min_, control_max_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void TubeMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0_actual) {
  this->state_.col(0) = x0_actual;
  state_array xdot;

  for (int i =0; i < this->num_timesteps_ - 1; ++i) {
    // Update the nominal state
    nominal_state_trajectory_.col(i + 1) = nominal_state_trajectory_.col(i);
    state_array state = nominal_state_trajectory_.col(i + 1);
    control_array control = nominal_control_trajectory_.col(i);
    this->model_->computeStateDeriv(state, control, xdot);
    this->model_->updateState(state, xdot, this->dt_);
    nominal_state_trajectory_.col(i + 1) = state;

    // Update the actual state
    this->state_.col(i + 1) = this->state_.col(i);
    state = this->state_.col(i + 1);
    control = this->control_.col(i);
    this->model_->computeStateDeriv(state, control, xdot);
    this->model_->updateState(state, xdot, this->dt_);
    this->state_.col(i + 1) = state;
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void TubeMPPI::updateNominalState(const Eigen::Ref<const control_array> &u) {
  state_array xdot;
  state_array state;
  this->model_->computeDynamics(nominal_state_trajectory_.col(0), u, xdot);
  this->model_->updateState(nominal_state_trajectory_.col(0), xdot, this->dt_);
}

