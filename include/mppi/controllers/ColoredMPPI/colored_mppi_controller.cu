#include <mppi/controllers/ColoredMPPI/colored_mppi_controller.cuh>
#include <mppi/core/mppi_common_new.cuh>
#include <mppi/core/mppi_common.cuh>
#include <algorithm>
#include <iostream>
#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh>

#define ColoredMPPI_TEMPLATE                                                                                           \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T>
#define ColoredMPPI ColoredMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

ColoredMPPI_TEMPLATE ColoredMPPI::ColoredMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller,
                                                        SAMPLING_T* sampler, float dt, int max_iter, float lambda,
                                                        float alpha, int num_timesteps,
                                                        const Eigen::Ref<const control_trajectory>& init_control_traj,
                                                        cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, init_control_traj,
                 stream)
{
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();
  // std::vector<float> tmp_vec(DYN_T::CONTROL_DIM, 0.0);
  // this->params_.colored_noise_exponents_ = std::move(tmp_vec);

  // Copy the noise std_dev to the device
  // this->copyControlStdDevToDevice();
}

ColoredMPPI_TEMPLATE ColoredMPPI::ColoredMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller,
                                                        SAMPLING_T* sampler, PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
{
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();
  // if (this->getColoredNoiseExponentsLValue().size() == 0)
  // {
  //   std::vector<float> tmp_vec(DYN_T::CONTROL_DIM, 0.0);
  //   getColoredNoiseExponentsLValue() = std::move(tmp_vec);
  // }

  // // Copy the noise std_dev to the device
  // this->copyControlStdDevToDevice();
}

ColoredMPPI_TEMPLATE ColoredMPPI::~ColoredMPPIController()
{
  // all implemented in standard controller
}

ColoredMPPI_TEMPLATE void ColoredMPPI::computeControl(const Eigen::Ref<const state_array>& state,
                                                      int optimization_stride)
{
  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost();
  state_array local_state = state;

  if (getLeashActive())
  {
    this->model_->enforceLeash(state, this->state_.col(leash_jump_), this->params_.state_leash_dist_, local_state);
  }

  // for testing convergence from scratch at every iteration
  // for (int i = optimization_stride + 1; i < MAX_TIMESTEPS; i++)
  // {
  //   this->control_.col(i) = this->control_.col(optimization_stride);
  // }

  // Send the initial condition to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, local_state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));

  float baseline_prev = 1e8;
  // control_array noise0 = this->getControlStdDev();
  for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++)
  {
    // Send the nominal control to the device
    this->copyNominalControlToDevice(false);

    // Generate noise data
    // const int colored_num_timesteps = (this->getNumTimesteps() > optimization_stride) ?
    //                                       this->getNumTimesteps() - optimization_stride :
    //                                       this->getNumTimesteps();
    // const int colored_stride = (this->getNumTimesteps() > optimization_stride) ? optimization_stride : 0;
    // if (colored_stride == 0)
    // {
    //   std::cout << "We tripped the fail-safe by having optimization stride greater than timestamps: "
    //             << optimization_stride << std::endl;
    // }
    // HANDLE_ERROR(cudaMemcpyAsync(this->control_d_, this->control_.data(),
    //                              sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM,
    //                              cudaMemcpyHostToDevice, this->stream_));
    // powerlaw_psd_gaussian(getColoredNoiseExponentsLValue(), this->getNumTimesteps(), NUM_ROLLOUTS,
    //                       this->control_noise_d_, optimization_stride, this->gen_, this->getOffsetDecayRate(),
    //                       this->stream_);
    // // scale noise down at each iteration
    // this->updateControlNoiseStdDev(noise0 * powf(control_std_dev_decay_, opt_iter));
    this->sampler_->generateSamples(optimization_stride, opt_iter, this->gen_, false);
    // Launch the rollout kernel
    // mppi_common::launchFastRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y, 1, COST_B_X, COST_B_Y>(
    //     this->model_->model_d_, this->cost_->cost_d_, this->getDt(), this->getNumTimesteps(), optimization_stride,
    //     this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->control_d_,
    //     this->control_noise_d_, this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_, false);
    mppi::kernels::launchFastRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_->model_d_, this->cost_->cost_d_, this->sampler_->sampling_d_, this->getDt(),
        this->getNumTimesteps(), NUM_ROLLOUTS, this->getLambda(), this->getAlpha(), this->initial_state_d_,
        this->output_d_, this->trajectory_costs_d_, this->params_.dynamics_rollout_dim_,
        this->params_.cost_rollout_dim_, this->stream_, false);

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
    if (getGamma() == 0 || getRExp() == 0)
    {
      mppi_common::launchNormExpKernel(NUM_ROLLOUTS, this->getNormExpThreads(), this->trajectory_costs_d_,
                                       1.0 / this->getLambda(), this->getBaselineCost(), this->stream_, false);
    }
    else
    {
      mppi_common::launchTsallisKernel(NUM_ROLLOUTS, this->getNormExpThreads(), this->trajectory_costs_d_, getGamma(),
                                       getRExp(), this->getBaselineCost(), this->stream_, false);
    }
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

    // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
    this->sampler_->updateDistributionParamsFromDevice(this->trajectory_costs_d_, this->getNormalizerCost(), 0, false);
    // mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
    //     this->trajectory_costs_d_, this->control_noise_d_, this->control_d_, this->getNormalizerCost(),
    //     this->getNumTimesteps(), this->stream_, false);

    // Transfer the new control to the host
    this->sampler_->setHostOptimalControlSequence(this->control_.data(), 0, true);
    // HANDLE_ERROR(cudaMemcpyAsync(this->control_.data(), this->control_d_,
    //                              sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM,
    //                              cudaMemcpyDeviceToHost, this->stream_));
    // HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
  // reset noise
  // this->updateControlNoiseStdDev(noise0);

  this->free_energy_statistics_.real_sys.normalizerPercent = this->getNormalizerCost() / NUM_ROLLOUTS;
  this->free_energy_statistics_.real_sys.increase =
      this->getBaselineCost() - this->free_energy_statistics_.real_sys.previousBaseline;
  smoothControlTrajectory();
  computeStateTrajectory(local_state);
  state_array zero_state = state_array::Zero();
  for (int i = 0; i < this->getNumTimesteps(); i++)
  {
    // this->model_->enforceConstraints(zero_state, this->control_.col(i));
    this->control_.col(i)[1] =
        fminf(fmaxf(this->control_.col(i)[1], this->model_->control_rngs_[1].x), this->model_->control_rngs_[1].y);
  }

  // Copy back sampled trajectories
  this->copySampledControlFromDevice(false);
  this->copyTopControlFromDevice(true);
}

ColoredMPPI_TEMPLATE void ColoredMPPI::allocateCUDAMemory()
{
  PARENT_CLASS::allocateCUDAMemoryHelper();
}

ColoredMPPI_TEMPLATE void ColoredMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0)
{
  this->computeOutputTrajectoryHelper(this->output_, this->state_, x0, this->control_);
}

ColoredMPPI_TEMPLATE void ColoredMPPI::slideControlSequence(int steps)
{
  // TODO does the logic of handling control history reasonable?
  leash_jump_ = steps;
  // Save the control history
  this->saveControlHistoryHelper(steps, this->control_, this->control_history_);

  this->slideControlSequenceHelper(steps, this->control_);
}

ColoredMPPI_TEMPLATE void ColoredMPPI::smoothControlTrajectory()
{
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
}

ColoredMPPI_TEMPLATE void ColoredMPPI::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->getTotalSampledTrajectories();
  // controls already copied in compute control
#if true
  mppi::kernels::launchVisualizeCostKernel<COST_T, SAMPLING_T>(
      this->cost_->cost_d_, this->sampler_->sampling_d_, this->getDt(), this->getNumTimesteps(),
      num_sampled_trajectories, this->getLambda(), this->getAlpha(), this->sampled_outputs_d_,
      this->sampled_crash_status_d_, this->sampled_costs_d_, this->params_.cost_rollout_dim_, this->vis_stream_, false);
#else
  mppi_common::launchVisualizeCostKernel<COST_T, 128, COST_B_Y, 1>(
      this->cost_->cost_d_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories, this->getLambda(),
      this->getAlpha(), this->sampled_outputs_d_, this->sampled_noise_d_, this->sampled_crash_status_d_,
      this->control_std_dev_d_, this->sampled_costs_d_, this->vis_stream_, false);
#endif
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
                                 this->getNumTimesteps() * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
}

#undef ColoredMPPI
