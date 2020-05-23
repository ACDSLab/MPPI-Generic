#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/core/mppi_common.cuh>
#include <algorithm>
#include <iostream>

#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost,
                                   float dt,
                                   int max_iter,
                                   float gamma,
                                   const Eigen::Ref<const control_array>& control_variance,
                                   int num_timesteps,
                                   const Eigen::Ref<const control_trajectory>& init_control_traj,
                                   cudaStream_t stream) :
Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(model, cost, dt, max_iter, gamma,
        control_variance, num_timesteps, init_control_traj, stream) {
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise variance to the device
  this->copyControlVarianceToDevice();

  // TODO copy the nominal trajectory from the first half to second half on device to set nominal the same as actual
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
VanillaMPPI::~VanillaMPPIController() {
  // all implemented in standard controller
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::computeControl(const Eigen::Ref<const state_array>& state) {

  // Send the initial condition to the device
  HANDLE_ERROR( cudaMemcpyAsync(this->initial_state_d_, state.data(),
      DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, this->stream_));

  float baseline_prev = 1e4;

  for (int opt_iter = 0; opt_iter < this->num_iters_; opt_iter++) {
    // Send the nominal control to the device
    this->copyNominalControlToDevice();

    //Generate noise data
    curandGenerateNormal(this->gen_, this->control_noise_d_,
                         NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM,
                         0.0, 1.0);

    //Launch the rollout kernel
    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
        this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_,
        this->initial_state_d_, this->control_d_, this->control_noise_d_,
        this->control_variance_d_, this->trajectory_costs_d_, this->stream_);


    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(),
        this->trajectory_costs_d_,
        NUM_ROLLOUTS*sizeof(float),
        cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR( cudaStreamSynchronize(this->stream_) );

    this->baseline_ = mppi_common::computeBaselineCost(this->trajectory_costs_.data(),
        NUM_ROLLOUTS);

    if (this->baseline_ > baseline_prev + 1) {
      // TODO handle printing
      std::cout << "Previous Baseline: " << baseline_prev << std::endl;
      std::cout << "         Baseline: " << this->baseline_ << std::endl;
    }

    baseline_prev = this->baseline_;

    // Launch the norm exponential kernel
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
        this->trajectory_costs_d_, this->gamma_, this->baseline_, this->stream_);
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(),
        this->trajectory_costs_d_,
        NUM_ROLLOUTS*sizeof(float),
        cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->normalizer_ = mppi_common::computeNormalizer(this->trajectory_costs_.data(),
        NUM_ROLLOUTS);

    // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
    mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
            this->trajectory_costs_d_, this->control_noise_d_, this->control_d_,
            this->normalizer_, this->num_timesteps_, this->stream_);

    // Transfer the new control to the host
    HANDLE_ERROR( cudaMemcpyAsync(this->control_.data(), this->control_d_,
            sizeof(float)*this->num_timesteps_*DYN_T::CONTROL_DIM,
            cudaMemcpyDeviceToHost, this->stream_));
    cudaStreamSynchronize(this->stream_);

    }
  smoothControlTrajectory();
  computeStateTrajectory(state);

}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::allocateCUDAMemory() {
  Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::allocateCUDAMemoryHelper();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0) {
  this->state_.col(0) = x0;
  state_array xdot;
  for (int i =0; i < this->num_timesteps_ - 1; ++i) {
    this->state_.col(i+1) = this->state_.col(i);
    state_array state = this->state_.col(i+1);
    control_array control = this->control_.col(i);
    this->model_->computeStateDeriv(state, control, xdot);
    this->model_->updateState(state, xdot, this->dt_);
    this->state_.col(i+1) = state;
    }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::slideControlSequence(int steps) {
  // TODO does the logic of handling control history reasonable?

  // Save the control history
  if (steps > 1) {
    this->control_history_.row(0) = this->control_.col(steps - 2).transpose();
    this->control_history_.row(1) = this->control_.col(steps - 1).transpose();
  } else { //
    this->control_history_.row(0) = this->control_history_.row(1); // Slide control history forward
    this->control_history_.row(1) = this->control_.col(0).transpose(); // Save the control at time 0
  }

  for (int i = 0; i < this->num_timesteps_; ++i) {
    for (int j = 0; j < DYN_T::CONTROL_DIM; j++) {
      int ind = std::min(i + steps, this->num_timesteps_ - 1);
      this->control_(j,i) = this->control_(j, ind);
    }
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::smoothControlTrajectory() {
  // TODO generalize to any size filter
  // TODO does the logic of handling control history reasonable?
  // Create the filter coefficients, pulled from table on wikipedia
  Eigen::Matrix<float, 1, 5> filter_coefficients;
  filter_coefficients << -3, 12, 17, 12, -3;
  filter_coefficients /= 35.0;

  // Create and fill a control buffer that we can apply the convolution filter
  Eigen::Matrix<float, MAX_TIMESTEPS+4, DYN_T::CONTROL_DIM> control_buffer;

  // Fill the first two timesteps with the control history
  control_buffer.topRows(2) = this->control_history_;

  // Fill the center timesteps with the current nominal trajectory
  control_buffer.middleRows(2, MAX_TIMESTEPS) = this->control_.transpose();

  // Fill the last two timesteps with the end of the current nominal control trajectory
  control_buffer.row(MAX_TIMESTEPS+2) = this->control_.transpose().row(MAX_TIMESTEPS-1);
  control_buffer.row(MAX_TIMESTEPS+3) = control_buffer.row(MAX_TIMESTEPS+2);

  // Apply convolutional filter to each timestep
  for (int i = 0; i < MAX_TIMESTEPS; ++i) {
    this->control_.col(i) = (filter_coefficients*control_buffer.middleRows(i,5)).transpose();
  }
}

#undef VanillaMPPI
