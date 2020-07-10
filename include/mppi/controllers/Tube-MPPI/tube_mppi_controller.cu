#include "tube_mppi_controller.cuh"


#define TubeMPPI TubeMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y>
TubeMPPI::TubeMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                             float lambda, float alpha,
                             const Eigen::Ref<const StateCostWeight>& Q,
                             const Eigen::Ref<const Hessian>& Qf,
                             const Eigen::Ref<const ControlCostWeight>& R,
                             const Eigen::Ref<const control_array>& control_std_dev,
                             int num_timesteps,
                             const Eigen::Ref<const control_trajectory>& init_control_traj,
                             cudaStream_t stream) :Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
                                     model, cost, dt, max_iter, lambda, alpha,
                                     control_std_dev, num_timesteps, init_control_traj, stream) {

  nominal_control_trajectory_ = init_control_traj;

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();

  // Initialize DDP
  this->initDDP(Q, Qf, R);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride) {
  if (!nominalStateInit_){
    // set the nominal state to the actual state
    nominal_state_trajectory_.col(0) = state;
    nominalStateInit_ = true;
  }

//  std::cout << "Post disturbance Actual State: "; this->model_->printState(state.data());
//  std::cout << "                Nominal State: "; this->model_->printState(nominal_state_trajectory_.col(0).data());

  // Handy reference pointers to the nominal state
  float * trajectory_costs_nominal_d = this->trajectory_costs_d_ + NUM_ROLLOUTS;
  float * initial_state_nominal_d = this->initial_state_d_ + DYN_T::STATE_DIM;

  float * control_noise_nominal_d = this->control_noise_d_ + NUM_ROLLOUTS * this->num_timesteps_ * DYN_T::CONTROL_DIM;
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
                         NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM*2,
                         0.0, 1.0);

    cudaDeviceSynchronize();

    // call rollout kernel with z = 2 since we have a nominal state
    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y, 2>(
            this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_,
            optimization_stride, this->lambda_, this->alpha_,
            this->initial_state_d_, this->control_d_, this->control_noise_d_,
            this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_);

    // Copy back sampled trajectories
    this->copySampledControlFromDevice();

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
        this->trajectory_costs_d_, 1.0/this->lambda_, this->baseline_, this->stream_);

    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
        trajectory_costs_nominal_d, 1.0/this->lambda_, this->baseline_nominal_, this->stream_);

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

    // TODO Find lambda and also add it to this method call
    mppi_common::computeFreeEnergy(this->free_energy_, this->free_energy_var_,
                                   this->trajectory_costs_.data(), NUM_ROLLOUTS,
                                   this->baseline_);

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
  this->saveControlHistoryHelper(steps, nominal_control_trajectory_, this->control_history_);

  this->slideControlSequenceHelper(steps, nominal_control_trajectory_);
  this->slideControlSequenceHelper(steps, this->control_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void TubeMPPI::smoothControlTrajectory() {
  this->smoothControlTrajectoryHelper(nominal_control_trajectory_, this->control_history_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void TubeMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0_actual) {
  // update the nominal state
  this->computeStateTrajectoryHelper(nominal_state_trajectory_, nominal_state_trajectory_.col(0), nominal_control_trajectory_);
  // update the actual state
  this->computeStateTrajectoryHelper(this->state_, x0_actual, this->control_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void TubeMPPI::updateNominalState(const Eigen::Ref<const control_array> &u) {
  state_array xdot;
  state_array state;
  this->model_->computeDynamics(nominal_state_trajectory_.col(0), u, xdot);
  this->model_->updateState(nominal_state_trajectory_.col(0), xdot, this->dt_);
}
