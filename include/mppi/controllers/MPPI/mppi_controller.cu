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
                                   float lambda,
                                   float alpha,
                                   const Eigen::Ref<const control_array>& control_std_dev,
                                   int num_timesteps,
                                   const Eigen::Ref<const control_trajectory>& init_control_traj,
                                   cudaStream_t stream) :
Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(model, cost, dt,
        max_iter, lambda, alpha, control_std_dev, num_timesteps, init_control_traj, stream) {
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
VanillaMPPI::~VanillaMPPIController() {
  // all implemented in standard controller
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride) {

  // Send the initial condition to the device
  HANDLE_ERROR( cudaMemcpyAsync(this->initial_state_d_, state.data(),
      DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, this->stream_));

  float baseline_prev = 1e8;

  for (int opt_iter = 0; opt_iter < this->num_iters_; opt_iter++) {
    // Send the nominal control to the device
    this->copyNominalControlToDevice();

    //Generate noise data
    curandGenerateNormal(this->gen_, this->control_noise_d_,
                         NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM,
                         0.0, 1.0);
    /*
    std::vector<float> noise = this->getSampledNoise();
    float mean = 0;
    for(int k = 0; k < noise.size(); k++) {
      mean += (noise[k]/noise.size());
    }

    float std_dev = 0;
    for(int k = 0; k < noise.size(); k++) {
      std_dev += powf(noise[k] - mean, 2);
    }
    std_dev = sqrt(std_dev/noise.size());
    printf("CPU 1 side N(%f, %f)\n", mean, std_dev);
     */

    //Launch the rollout kernel
    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
        this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_,
        optimization_stride, this->lambda_, this->alpha_,
        this->initial_state_d_, this->control_d_, this->control_noise_d_,
        this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_);
    /*
    noise = this->getSampledNoise();
    mean = 0;
    for(int k = 0; k < noise.size(); k++) {
      mean += (noise[k]/noise.size());
    }

    std_dev = 0;
    for(int k = 0; k < noise.size(); k++) {
      std_dev += powf(noise[k] - mean, 2);
    }
    std_dev = sqrt(std_dev/noise.size());
    printf("CPU 2 side N(%f, %f)\n", mean, std_dev);
     */

    // Copy back sampled trajectories
    this->copySampledControlFromDevice();
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
      if (this->debug_) {
        std::cout << "Previous Baseline: " << baseline_prev << std::endl;
        std::cout << "         Baseline: " << this->baseline_ << std::endl;
      }

    }

    baseline_prev = this->baseline_;

    // Launch the norm exponential kernel
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
        this->trajectory_costs_d_, 1.0/this->lambda_, this->baseline_, this->stream_);
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(),
        this->trajectory_costs_d_,
        NUM_ROLLOUTS*sizeof(float),
        cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->normalizer_ = mppi_common::computeNormalizer(this->trajectory_costs_.data(),
        NUM_ROLLOUTS);

    // TODO Find lambda and also add it to this method call
    mppi_common::computeFreeEnergy(this->free_energy_, this->free_energy_var_,
                                   this->trajectory_costs_.data(), NUM_ROLLOUTS,
                                   this->baseline_);

    // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
    mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
            this->trajectory_costs_d_, this->control_noise_d_, this->control_d_,
            this->normalizer_, this->num_timesteps_, this->stream_);

    /*
    noise = this->getSampledNoise();
    mean = 0;
    for(int k = 0; k < noise.size(); k++) {
      mean += (noise[k]/noise.size());
    }

    std_dev = 0;
    for(int k = 0; k < noise.size(); k++) {
      std_dev += powf(noise[k] - mean, 2);
    }
    std_dev = sqrt(std_dev/noise.size());
    printf("CPU 3 side N(%f, %f)\n", mean, std_dev);
     */

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
  this->computeStateTrajectoryHelper(this->state_, x0, this->control_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::slideControlSequence(int steps) {
  // TODO does the logic of handling control history reasonable?

  // Save the control history
  this->saveControlHistoryHelper(steps, this->control_, this->control_history_);

  this->slideControlSequenceHelper(steps, this->control_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::smoothControlTrajectory() {
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
}

#undef VanillaMPPI
