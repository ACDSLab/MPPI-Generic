#include "tube_mppi_controller.cuh"

#define TubeMPPI TubeMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y, PARAMS_T>

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
TubeMPPI::TubeMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, float dt, int max_iter, float lambda,
                             float alpha, const Eigen::Ref<const control_array>& control_std_dev, int num_timesteps,
                             const Eigen::Ref<const control_trajectory>& init_control_traj, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, dt, max_iter, lambda, alpha, control_std_dev, num_timesteps,
                 init_control_traj, stream)
{
  nominal_control_trajectory_ = init_control_traj;

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();

  // Initialize Feedback
  this->fb_controller_->initTrackingController();
  this->enable_feedback_ = true;
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
TubeMPPI::TubeMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, params, stream)
{
  nominal_control_trajectory_ = this->params_.init_control_traj_;

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();

  // Initialize Feedback
  this->fb_controller_->initTrackingController();
  this->enable_feedback_ = true;
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  if (!nominalStateInit_)
  {
    // set the nominal state to the actual state
    nominal_state_trajectory_.col(0) = state;
    nominalStateInit_ = true;
  }

  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost(0);
  this->free_energy_statistics_.nominal_sys.previousBaseline = this->getBaselineCost(1);

  //  std::cout << "Post disturbance Actual State: "; this->model_->printState(state.data());
  //  std::cout << "                Nominal State: "; this->model_->printState(nominal_state_trajectory_.col(0).data());

  // Handy reference pointers to the nominal state
  float* trajectory_costs_nominal_d = this->trajectory_costs_d_ + NUM_ROLLOUTS;
  float* initial_state_nominal_d = this->initial_state_d_ + DYN_T::STATE_DIM;

  float* control_noise_nominal_d = this->control_noise_d_ + NUM_ROLLOUTS * this->getNumTimesteps() * DYN_T::CONTROL_DIM;
  float* control_nominal_d = this->control_d_ + this->getNumTimesteps() * DYN_T::CONTROL_DIM;

  for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++)
  {
    // Send the initial condition to the device
    HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), DYN_T::STATE_DIM * sizeof(float),
                                 cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(initial_state_nominal_d, nominal_state_trajectory_.data(),
                                 DYN_T::STATE_DIM * sizeof(float), cudaMemcpyHostToDevice, this->stream_));

    // Send the nominal control to the device
    copyControlToDevice(false);

    // Generate noise data
    curandGenerateNormal(this->gen_, this->control_noise_d_,
                         NUM_ROLLOUTS * this->getNumTimesteps() * DYN_T::CONTROL_DIM, 0.0, 1.0);

    HANDLE_ERROR(cudaMemcpyAsync(control_noise_nominal_d, this->control_noise_d_,
                                 NUM_ROLLOUTS * this->getNumTimesteps() * DYN_T::CONTROL_DIM * sizeof(float),
                                 cudaMemcpyDeviceToDevice, this->stream_));

    // call rollout kernel with z = 2 since we have a nominal state
    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y, 2>(
        this->model_->model_d_, this->cost_->cost_d_, this->getDt(), this->getNumTimesteps(), optimization_stride,
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->control_d_, this->control_noise_d_,
        this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_, false);

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));

    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(), trajectory_costs_nominal_d,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    this->setBaseline(mppi_common::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS), 0);
    this->setBaseline(mppi_common::computeBaselineCost(this->trajectory_costs_nominal_.data(), NUM_ROLLOUTS), 1);

    // Launch the norm exponential kernel for both actual and nominal
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X, this->trajectory_costs_d_, 1.0 / this->getLambda(),
                                     this->getBaselineCost(0), this->stream_, false);

    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X, trajectory_costs_nominal_d, 1.0 / this->getLambda(),
                                     this->getBaselineCost(1), this->stream_, false);

    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(), trajectory_costs_nominal_d,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->setNormalizer(mppi_common::computeNormalizer(this->trajectory_costs_.data(), NUM_ROLLOUTS), 0);
    this->setNormalizer(mppi_common::computeNormalizer(this->trajectory_costs_nominal_.data(), NUM_ROLLOUTS), 1);

    // Compute real free energy
    mppi_common::computeFreeEnergy(this->free_energy_statistics_.real_sys.freeEnergyMean,
                                   this->free_energy_statistics_.real_sys.freeEnergyVariance,
                                   this->free_energy_statistics_.real_sys.freeEnergyModifiedVariance,
                                   this->trajectory_costs_.data(), NUM_ROLLOUTS, this->getBaselineCost(0),
                                   this->getLambda());

    // Compute Nominal State free Energy
    mppi_common::computeFreeEnergy(this->free_energy_statistics_.nominal_sys.freeEnergyMean,
                                   this->free_energy_statistics_.nominal_sys.freeEnergyVariance,
                                   this->free_energy_statistics_.nominal_sys.freeEnergyModifiedVariance,
                                   this->trajectory_costs_nominal_.data(), NUM_ROLLOUTS, this->getBaselineCost(1),
                                   this->getLambda());

    // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
    mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
        this->trajectory_costs_d_, this->control_noise_d_, this->control_d_, this->getNormalizerCost(0),
        this->getNumTimesteps(), this->stream_, false);
    mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
        trajectory_costs_nominal_d, control_noise_nominal_d, control_nominal_d, this->getNormalizerCost(1),
        this->getNumTimesteps(), this->stream_, false);

    // Transfer the new control to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->control_.data(), this->control_d_,
                                 sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToHost,
                                 this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(nominal_control_trajectory_.data(), control_nominal_d,
                                 sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToHost,
                                 this->stream_));
    cudaStreamSynchronize(this->stream_);

    // Compute the nominal and actual state trajectories
    computeStateTrajectory(state);  // Input is the actual state

    //    std::cout << "Actual baseline: " << this->getBaselineCost(0) << std::endl;
    //    std::cout << "Nominal baseline: " << this->getBaselineCost(1) << std::endl;

    if (this->getBaselineCost(0) < this->getBaselineCost(1) + getNominalThreshold())
    {
      // In this case, the disturbance the made the nominal and actual states differ improved the cost.
      // std::copy(state_trajectory.begin(), state_trajectory.end(), nominal_state_trajectory_.begin());
      // std::copy(control_trajectory.begin(), control_trajectory.end(), nominal_control_.begin());
      this->free_energy_statistics_.nominal_state_used = 0;
      nominal_state_trajectory_ = this->state_;
      nominal_control_trajectory_ = this->control_;
    }
    else
    {
      this->free_energy_statistics_.nominal_state_used = 1;
    }

    // Outside of this loop, we will utilize the nominal state trajectory and the nominal control trajectory to compute
    // the optimal feedback gains using our ancillary controller, then apply feedback inside our main while loop at the
    // same rate as our state estimator.
  }
  smoothControlTrajectory();
  computeStateTrajectory(state);  // Input is the actual state

  this->free_energy_statistics_.real_sys.normalizerPercent = this->getNormalizerCost(0) / NUM_ROLLOUTS;
  this->free_energy_statistics_.real_sys.increase =
      this->getBaselineCost(0) - this->free_energy_statistics_.real_sys.previousBaseline;
  this->free_energy_statistics_.nominal_sys.normalizerPercent = this->getNormalizerCost(1) / NUM_ROLLOUTS;
  this->free_energy_statistics_.nominal_sys.increase =
      this->getBaselineCost(1) - this->free_energy_statistics_.nominal_sys.previousBaseline;

  // Copy back sampled trajectories
  this->copySampledControlFromDevice(false);
  this->copyTopControlFromDevice(true);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::copyControlToDevice(bool synchronize)
{
  HANDLE_ERROR(cudaMemcpyAsync(this->control_d_, this->control_.data(), sizeof(float) * this->control_.size(),
                               cudaMemcpyHostToDevice, this->stream_));

  HANDLE_ERROR(cudaMemcpyAsync(this->control_d_ + this->control_.size(), nominal_control_trajectory_.data(),
                               sizeof(float) * nominal_control_trajectory_.size(), cudaMemcpyHostToDevice,
                               this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::allocateCUDAMemory()
{
  PARENT_CLASS::allocateCUDAMemoryHelper(1);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::slideControlSequence(int steps)
{
  // Propagate the nominal trajectory forward
  updateNominalState(nominal_control_trajectory_.col(0));

  // Save the control history
  this->saveControlHistoryHelper(steps, nominal_control_trajectory_, this->control_history_);

  this->slideControlSequenceHelper(steps, nominal_control_trajectory_);
  this->slideControlSequenceHelper(steps, this->control_);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::smoothControlTrajectory()
{
  this->smoothControlTrajectoryHelper(nominal_control_trajectory_, this->control_history_);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0_actual)
{
  // update the nominal state
  this->computeStateTrajectoryHelper(nominal_state_trajectory_, nominal_state_trajectory_.col(0),
                                     nominal_control_trajectory_);
  // update the actual state
  this->computeStateTrajectoryHelper(this->state_, x0_actual, this->control_);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::updateNominalState(const Eigen::Ref<const control_array>& u)
{
  state_array xdot;
  output_array output;
  this->model_->step(nominal_state_trajectory_.col(0), nominal_state_trajectory_.col(0), xdot, u, output, 0,
                     this->getDt());
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T>
void TubeMPPI::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->getTotalSampledTrajectories();
  // control already copied in compute control, so run kernel
  mppi_common::launchStateAndCostTrajectoryKernel<DYN_T, COST_T, FEEDBACK_GPU, BDIM_X, BDIM_Y, 2>(
      this->model_->model_d_, this->cost_->cost_d_, this->fb_controller_->getDevicePointer(), this->sampled_noise_d_,
      this->initial_state_d_, this->sampled_outputs_d_, this->sampled_costs_d_, this->sampled_crash_status_d_,
      num_sampled_trajectories, this->getNumTimesteps(), this->getDt(), this->vis_stream_);

  // copy back results
  for (int i = 0; i < num_sampled_trajectories * 2; i++)
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_trajectories_[i].data(),
                                 this->sampled_outputs_d_ + i * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
                                 this->getNumTimesteps() * DYN_T::OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost,
                                 this->vis_stream_));
  }
  HANDLE_ERROR(cudaMemcpyAsync(this->sampled_costs_.data(), this->sampled_costs_d_,
                               this->getNumTimesteps() * 2 * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
  HANDLE_ERROR(cudaMemcpyAsync(this->sampled_crash_status_.data(), this->sampled_crash_status_d_,
                               this->getNumTimesteps() * 2 * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
  HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
}
