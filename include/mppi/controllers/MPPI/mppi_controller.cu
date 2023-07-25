#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/core/mppi_common_new.cuh>
#include <mppi/core/mppi_common.cuh>
#include <algorithm>
#include <iostream>

#define VANILLA_MPPI_TEMPLATE                                                                                          \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T>

#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

VANILLA_MPPI_TEMPLATE
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt,
                                   int max_iter, float lambda, float alpha, int num_timesteps,
                                   const Eigen::Ref<const control_trajectory>& init_control_traj, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, init_control_traj,
                 stream)
{
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  // this->copyControlStdDevToDevice();
}

VANILLA_MPPI_TEMPLATE
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler,
                                   PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
{
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // // Copy the noise std_dev to the device
  // this->copyControlStdDevToDevice();
}

VANILLA_MPPI_TEMPLATE
VanillaMPPI::~VanillaMPPIController()
{
  // all implemented in standard controller
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost();

  // Send the initial condition to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));

  float baseline_prev = 1e8;

  for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++)
  {
    // Send the nominal control to the device
    this->copyNominalControlToDevice(false);

    // Generate noise data
    this->sampler_->generateSamples(optimization_stride, opt_iter, this->gen_, false);
    // curandGenerateNormal(this->gen_, this->control_noise_d_,
    //                      NUM_ROLLOUTS * this->getNumTimesteps() * DYN_T::CONTROL_DIM, 0.0, 1.0);
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

    // Launch the rollout kernel
    // mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
    //     this->model_->model_d_, this->cost_->cost_d_, this->getDt(), this->getNumTimesteps(), optimization_stride,
    //     this->getLambda(), this->getAlpha(), this->initial_state_d_, this->control_d_, this->control_noise_d_,
    //     this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_, false);
    // mppi_common::launchFastRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y, 1, COST_B_X, COST_B_Y>(
    //     this->model_->model_d_, this->cost_->cost_d_, this->getDt(), this->getNumTimesteps(), optimization_stride,
    //     this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->control_d_,
    //     this->control_noise_d_, this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_, false);
    mppi::kernels::launchFastRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_->model_d_, this->cost_->cost_d_, this->sampler_->sampling_d_, this->getDt(),
        this->getNumTimesteps(), NUM_ROLLOUTS, this->getLambda(), this->getAlpha(), this->initial_state_d_,
        this->output_d_, this->trajectory_costs_d_, this->params_.dynamics_rollout_dim_,
        this->params_.cost_rollout_dim_, this->stream_, false);
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

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    this->setBaseline(mppi_common::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS));

    if (this->getBaselineCost() > baseline_prev + 1)
    {
      // TODO handle printing
      if (this->debug_)
      {
        std::cout << "Previous Baseline: " << baseline_prev << std::endl;
        std::cout << "         Baseline: " << this->getBaselineCost() << std::endl;
      }
    }

    baseline_prev = this->getBaselineCost();

    // Launch the norm exponential kernel
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, this->getNormExpThreads(), this->trajectory_costs_d_,
                                     1.0 / this->getLambda(), this->getBaselineCost(), this->stream_, false);
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->setNormalizer(mppi_common::computeNormalizer(this->trajectory_costs_.data(), NUM_ROLLOUTS));

    mppi_common::computeFreeEnergy(this->free_energy_statistics_.real_sys.freeEnergyMean,
                                   this->free_energy_statistics_.real_sys.freeEnergyVariance,
                                   this->free_energy_statistics_.real_sys.freeEnergyModifiedVariance,
                                   this->trajectory_costs_.data(), NUM_ROLLOUTS, this->getBaselineCost(),
                                   this->getLambda());

    this->sampler_->updateDistributionParamsFromDevice(this->trajectory_costs_d_, this->getNormalizerCost(), 0, false);

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
    this->sampler_->setHostOptimalControlSequence(this->control_.data(), 0, true);
    // HANDLE_ERROR(cudaMemcpyAsync(this->control_.data(), this->control_d_,
    //                              sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM,
    //                              cudaMemcpyDeviceToHost, this->stream_));
    // HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }

  this->free_energy_statistics_.real_sys.normalizerPercent = this->getNormalizerCost() / NUM_ROLLOUTS;
  this->free_energy_statistics_.real_sys.increase =
      this->getBaselineCost() - this->free_energy_statistics_.real_sys.previousBaseline;
  smoothControlTrajectory();
  computeStateTrajectory(state);
  state_array zero_state = state_array::Zero();
  for (int i = 0; i < this->getNumTimesteps(); i++)
  {
    this->model_->enforceConstraints(zero_state, this->control_.col(i));
  }

  // Copy back sampled trajectories
  this->copySampledControlFromDevice(false);
  this->copyTopControlFromDevice(true);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::allocateCUDAMemory()
{
  PARENT_CLASS::allocateCUDAMemoryHelper();
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0)
{
  this->computeOutputTrajectoryHelper(this->output_, this->state_, x0, this->control_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::slideControlSequence(int steps)
{
  // TODO does the logic of handling control history reasonable?

  // Save the control history
  this->saveControlHistoryHelper(steps, this->control_, this->control_history_);

  this->slideControlSequenceHelper(steps, this->control_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::smoothControlTrajectory()
{
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->getTotalSampledTrajectories();

  // control already copied in compute control, so run kernel
  mppi::kernels::launchVisualizeCostKernel<COST_T, SAMPLING_T>(
      this->cost_->cost_d_, this->sampler_->sampling_d_, this->getDt(), this->getNumTimesteps(),
      num_sampled_trajectories, this->getLambda(), this->getAlpha(), this->sampled_outputs_d_,
      this->sampled_crash_status_d_, this->sampled_costs_d_, this->params_.cost_rollout_dim_, this->stream_, false);
  // #if true
  //   mppi_common::launchVisualizeCostKernel<COST_T, 128, 4, 1>(
  //       this->cost_->cost_d_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories, this->getLambda(),
  //       this->getAlpha(), this->sampled_outputs_d_, this->sampled_noise_d_, this->sampled_crash_status_d_,
  //       this->control_std_dev_d_, this->sampled_costs_d_, this->vis_stream_, false);
  // #else
  //   mppi_common::launchStateAndCostTrajectoryKernel<DYN_T, COST_T, FEEDBACK_GPU, BDIM_X, BDIM_Y>(
  //       this->model_->model_d_, this->cost_->cost_d_, this->fb_controller_->getDevicePointer(),
  //       this->sampled_noise_d_, this->initial_state_d_, this->sampled_outputs_d_, this->sampled_costs_d_,
  //       this->sampled_crash_status_d_, num_sampled_trajectories, this->getNumTimesteps(), this->getDt(),
  //       this->vis_stream_);
  // #endif

  for (int i = 0; i < num_sampled_trajectories; i++)
  {
    // set initial state to the first location
    // shifted by one since we do not save the initial state
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_trajectories_[i].data(),
                                 this->sampled_outputs_d_ + i * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
                                 (this->getNumTimesteps() - 1) * DYN_T::OUTPUT_DIM * sizeof(float),
                                 cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(
        cudaMemcpyAsync(this->sampled_costs_[i].data(), this->sampled_costs_d_ + (i * (this->getNumTimesteps() + 1)),
                        (this->getNumTimesteps() + 1) * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_crash_status_[i].data(),
                                 this->sampled_crash_status_d_ + (i * this->getNumTimesteps()),
                                 this->getNumTimesteps() * sizeof(int), cudaMemcpyDeviceToHost, this->vis_stream_));
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
}

#undef VANILLA_MPPI_TEMPLATE
#undef VanillaMPPI
