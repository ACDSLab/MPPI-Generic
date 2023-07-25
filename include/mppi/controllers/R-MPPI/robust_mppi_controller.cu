#include "robust_mppi_controller.cuh"
#include <Eigen/Eigenvalues>
#include <exception>

#define ROBUST_MPPI_TEMPLATE                                                                                           \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T, int SAMPLES_PER_CONDITION_MULTIPLIER>

#define RobustMPPI                                                                                                     \
  RobustMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T,                         \
                       SAMPLES_PER_CONDITION_MULTIPLIER>

ROBUST_MPPI_TEMPLATE
RobustMPPI::RobustMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, float dt, int max_iter, float lambda,
                                 float alpha, float value_function_threshold,
                                 const Eigen::Ref<const control_array>& control_std_dev, int num_timesteps,
                                 const Eigen::Ref<const control_trajectory>& init_control_traj,
                                 int num_candidate_nominal_states, int optimization_stride, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, dt, max_iter, lambda, alpha, control_std_dev, num_timesteps,
                 init_control_traj, stream)
{
  setValueFunctionThreshold(value_function_threshold);
  setOptimizationStride(optimization_stride);
  setNumCandidates(num_candidate_nominal_states);
  updateNumCandidates(getNumCandidates());

  // Zero the control history
  this->control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();
  nominal_control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();

  // Initialize Feedback
  this->fb_controller_->initTrackingController();

  // Initialize the nominal control trajectory
  nominal_control_trajectory_ = init_control_traj;

  this->enable_feedback_ = true;
}

ROBUST_MPPI_TEMPLATE
RobustMPPI::RobustMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, params, stream)
{
  updateNumCandidates(getNumCandidates());

  // Zero the control history
  this->control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();
  nominal_control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();

  // Initialize Feedback
  this->fb_controller_->initTrackingController();

  // Initialize the nominal control trajectory
  nominal_control_trajectory_ = this->params_.init_control_traj_;

  this->enable_feedback_ = true;
}

ROBUST_MPPI_TEMPLATE
RobustMPPI::~RobustMPPIController()
{
  deallocateNominalStateCandidateMemory();
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::allocateCUDAMemory()
{
  PARENT_CLASS::allocateCUDAMemoryHelper(1);
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::getInitNominalStateCandidates(const Eigen::Ref<const state_array>& nominal_x_k,
                                               const Eigen::Ref<const state_array>& nominal_x_kp1,
                                               const Eigen::Ref<const state_array>& real_x_kp1)
{
  Eigen::MatrixXf points(DYN_T::STATE_DIM, 3);
  points << nominal_x_k, nominal_x_kp1, real_x_kp1;
  auto candidates = points * line_search_weights_;
  for (int i = 0; i < getNumCandidates(); ++i)
  {
    candidate_nominal_states_[i] = candidates.col(i);
  }
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::resetCandidateCudaMem()
{
  deallocateNominalStateCandidateMemory();
  HANDLE_ERROR(
      cudaMalloc((void**)&importance_sampling_states_d_, sizeof(float) * DYN_T::STATE_DIM * getNumCandidates()));
  HANDLE_ERROR(
      cudaMalloc((void**)&importance_sampling_costs_d_, sizeof(float) * getNumCandidates() * SAMPLES_PER_CONDITION));
  HANDLE_ERROR(cudaMalloc((void**)&importance_sampling_strides_d_, sizeof(int) * getNumCandidates()));

  // Set flag so that the we know cudamemory is allocated
  importance_sampling_cuda_mem_init_ = true;
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::deallocateNominalStateCandidateMemory()
{
  if (importance_sampling_cuda_mem_init_)
  {
    HANDLE_ERROR(cudaFree(importance_sampling_states_d_));
    HANDLE_ERROR(cudaFree(importance_sampling_costs_d_));
    HANDLE_ERROR(cudaFree(importance_sampling_strides_d_));

    // Set flag so that we know cudamemory has been freed
    importance_sampling_cuda_mem_init_ = false;
  }
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::copyNominalControlToDevice(bool synchronize)
{
  HANDLE_ERROR(cudaMemcpyAsync(this->control_d_, nominal_control_trajectory_.data(),
                               sizeof(float) * nominal_control_trajectory_.size(), cudaMemcpyHostToDevice,
                               this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::updateNumCandidates(int new_num_candidates)
{
  if ((new_num_candidates * SAMPLES_PER_CONDITION) > NUM_ROLLOUTS)
  {
    std::cerr << "ERROR: (number of candidates) * (SAMPLES_PER_CONDITION) cannot exceed NUM_ROLLOUTS\n";
    std::cerr << "number of candidates: " << new_num_candidates << ", SAMPLES_PER_CONDITION: " << SAMPLES_PER_CONDITION
              << ", NUM_ROLLOUTS: " << NUM_ROLLOUTS << "\n";
    std::terminate();
  }

  // New number must be odd and greater than 3
  if (new_num_candidates < 3)
  {
    std::cerr << "ERROR: number of candidates must be greater or equal to 3\n";
    std::cerr << "number of candidates: " << new_num_candidates << "\n";
    std::terminate();
  }
  if (new_num_candidates % 2 == 0)
  {
    std::cerr << "ERROR: number of candidates must be odd\n";
    std::cerr << "number of candidates: " << new_num_candidates << "\n";
    std::terminate();
  }

  // Set the new value of the number of candidates
  setNumCandidates(new_num_candidates);

  // Resize the vector holding the candidate nominal states
  candidate_nominal_states_.resize(getNumCandidates());

  // Resize the matrix holding the importance sampler strides
  importance_sampler_strides_.resize(1, getNumCandidates());

  // Resize the trajectory costs matrix
  candidate_trajectory_costs_.resize(getNumCandidates() * SAMPLES_PER_CONDITION, 1);
  candidate_trajectory_costs_.setZero();

  // Resize the free energy costs matrix
  candidate_free_energy_.resize(getNumCandidates(), 1);
  candidate_free_energy_.setZero();

  // Deallocate and reallocate cuda memory
  resetCandidateCudaMem();

  // Recompute the line search weights based on the number of candidates
  computeLineSearchWeights();
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::computeLineSearchWeights()
{
  line_search_weights_.resize(3, getNumCandidates());

  // For a given setup, this never changes.... why recompute every time?
  int num_candid_over_2 = getNumCandidates() / 2;
  for (int i = 0; i < num_candid_over_2 + 1; i++)
  {
    line_search_weights_(0, i) = 1 - i / float(num_candid_over_2);
    line_search_weights_(1, i) = i / float(num_candid_over_2);
    line_search_weights_(2, i) = 0.0;
  }
  for (int i = 1; i < num_candid_over_2 + 1; i++)
  {
    line_search_weights_(0, num_candid_over_2 + i) = 0.0;
    line_search_weights_(1, num_candid_over_2 + i) = 1 - i / float(num_candid_over_2);
    line_search_weights_(2, num_candid_over_2 + i) = i / float(num_candid_over_2);
  }
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::computeImportanceSamplerStride(int stride)
{
  Eigen::MatrixXf stride_vec(1, 3);
  stride_vec << 0, stride, stride;

  // Perform matrix multiplication, convert to array so that we can round the floats to the nearest
  // integer. Then cast the resultant float array to an int array. Then set equal to our int matrix.
  importance_sampler_strides_ = (stride_vec * line_search_weights_).array().round().template cast<int>();
}

ROBUST_MPPI_TEMPLATE
float RobustMPPI::computeCandidateBaseline()
{
  float baseline = candidate_trajectory_costs_(0);
  for (int i = 0; i < SAMPLES_PER_CONDITION * getNumCandidates(); i++)
  {  // TODO What is the reasoning behind only using the first condition to get the baseline?
    if (candidate_trajectory_costs_(i) < baseline)
    {
      baseline = candidate_trajectory_costs_(i);
    }
  }
  return baseline;
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::computeBestIndex()
{
  candidate_free_energy_.setZero();
  float baseline = computeCandidateBaseline();
  for (int i = 0; i < getNumCandidates(); i++)
  {
    for (int j = 0; j < SAMPLES_PER_CONDITION; j++)
    {
      candidate_free_energy_(i) +=
          expf(-1.0 / this->getLambda() * (candidate_trajectory_costs_(i * SAMPLES_PER_CONDITION + j) - baseline));
    }
    candidate_free_energy_(i) /= (1.0 * SAMPLES_PER_CONDITION);
    candidate_free_energy_(i) = -this->getLambda() * logf(candidate_free_energy_(i)) + baseline;
    if (candidate_free_energy_(i) < getValueFunctionThreshold())
    {
      best_index_ = i;
    }
  }
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::updateImportanceSamplingControl(const Eigen::Ref<const state_array>& state, int stride)
{
  // (Controller Frequency)*(Optimization Time) corresponds to how many timesteps occurred in the last optimization
  real_stride_ = stride;

  computeNominalStateAndStride(state, stride);  // Launches the init eval kernel

  // Save the nominal control history for the importance sampler
  this->saveControlHistoryHelper(nominal_stride_, nominal_control_trajectory_, nominal_control_history_);

  // Save the real control history for the optimal control
  this->saveControlHistoryHelper(real_stride_, this->control_, this->control_history_);

  // Slide the control sequence for the nominal control trajectory
  this->slideControlSequenceHelper(nominal_stride_, nominal_control_trajectory_);

  // Compute the nominal trajectory because we have slid the control sequence and updated the nominal state
  this->computeStateTrajectoryHelper(nominal_state_trajectory_, nominal_state_, nominal_control_trajectory_);

  // Compute the feedback gains and save them to an array
  computeNominalFeedbackGains(state);
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::computeNominalStateAndStride(const Eigen::Ref<const state_array>& state, int stride)
{
  if (!nominal_state_init_)
  {
    nominal_state_ = state;
    nominal_state_init_ = true;
    nominal_stride_ = 0;
  }
  else
  {
    getInitNominalStateCandidates(nominal_state_trajectory_.col(0), nominal_state_trajectory_.col(1), state);
    computeImportanceSamplerStride(stride);

    // Send the nominal state candidates to the GPU
    HANDLE_ERROR(cudaMemcpyAsync(importance_sampling_states_d_, candidate_nominal_states_.data(),
                                 sizeof(float) * DYN_T::STATE_DIM * getNumCandidates(), cudaMemcpyHostToDevice,
                                 this->stream_));
    // Send the importance sampler strides to the GPU
    HANDLE_ERROR(cudaMemcpyAsync(importance_sampling_strides_d_, importance_sampler_strides_.data(),
                                 sizeof(int) * getNumCandidates(), cudaMemcpyHostToDevice, this->stream_));

    // Send the nominal control to the GPU
    copyNominalControlToDevice(false);

    // Generate noise for the samples
    HANDLE_CURAND_ERROR(curandGenerateNormal(
        this->gen_, this->control_noise_d_,
        SAMPLES_PER_CONDITION * getNumCandidates() * this->getNumTimesteps() * DYN_T::CONTROL_DIM, 0.0, 1.0));

    // Launch the init eval kernel
    rmppi_kernels::launchInitEvalKernel<DYN_T, COST_T, BDIM_X, BDIM_Y, SAMPLES_PER_CONDITION>(
        this->model_->model_d_, this->cost_->cost_d_, getNumCandidates(), this->getNumTimesteps(), this->getLambda(),
        this->getAlpha(), stride, this->getDt(), importance_sampling_strides_d_, this->control_std_dev_d_,
        importance_sampling_states_d_, this->control_d_, this->control_noise_d_, importance_sampling_costs_d_,
        this->stream_, false);

    HANDLE_ERROR(cudaMemcpyAsync(candidate_trajectory_costs_.data(), importance_sampling_costs_d_,
                                 sizeof(float) * getNumCandidates() * SAMPLES_PER_CONDITION, cudaMemcpyDeviceToHost,
                                 this->stream_));
    cudaStreamSynchronize(this->stream_);

    // Compute the best nominal state candidate from the rollouts
    computeBestIndex();
    //    best_index_ = 8;
    this->free_energy_statistics_.nominal_state_used = best_index_;
    nominal_stride_ = importance_sampler_strides_(best_index_);
    nominal_state_ = candidate_nominal_states_[best_index_];
  }
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::computeNominalFeedbackGains(const Eigen::Ref<const state_array>& state)
{
  this->computeFeedbackHelper(state, nominal_state_trajectory_, nominal_control_trajectory_);
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  // Handy dandy pointers to nominal data
  float* trajectory_costs_nominal_d = this->trajectory_costs_d_ + NUM_ROLLOUTS;
  float* initial_state_nominal_d = this->initial_state_d_ + DYN_T::STATE_DIM;
  float* control_noise_nominal_d = this->control_noise_d_ + NUM_ROLLOUTS * this->getNumTimesteps() * DYN_T::CONTROL_DIM;
  float* control_nominal_d = this->control_d_ + this->getNumTimesteps() * DYN_T::CONTROL_DIM;

  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost(0);
  this->free_energy_statistics_.nominal_sys.previousBaseline = this->getBaselineCost(1);

  // Transfer the feedback gains to the GPU
  this->fb_controller_->copyToDevice(false);

  // Transfer the real initial state to the GPU
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, this->stream_));
  // Transfer the nominal state to the GPU: recall that the device GPU has the augmented state [real state, nominal
  // state]
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_nominal_d, nominal_state_.data(), sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, this->stream_));

  for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++)
  {
    // Copy the importance sampling control to the system
    HANDLE_ERROR(cudaMemcpyAsync(this->control_d_, nominal_control_trajectory_.data(),
                                 sizeof(float) * DYN_T::CONTROL_DIM * this->getNumTimesteps(), cudaMemcpyHostToDevice,
                                 this->stream_));

    // Generate a the control perturbations for exploration
    curandGenerateNormal(this->gen_, this->control_noise_d_,
                         DYN_T::CONTROL_DIM * this->getNumTimesteps() * NUM_ROLLOUTS, 0.0, 1.0);

    // Make a second copy of the random deviations
    HANDLE_ERROR(cudaMemcpyAsync(control_noise_nominal_d, this->control_noise_d_,
                                 DYN_T::CONTROL_DIM * this->getNumTimesteps() * NUM_ROLLOUTS * sizeof(float),
                                 cudaMemcpyDeviceToDevice, this->stream_));

    // Launch the new rollout kernel
    rmppi_kernels::launchRMPPIRolloutKernel<DYN_T, COST_T, FEEDBACK_GPU, NUM_ROLLOUTS, BLOCKSIZE_X, BLOCKSIZE_Y, 2>(
        this->model_->model_d_, this->cost_->cost_d_, this->fb_controller_->getDevicePointer(), this->getDt(),
        this->getNumTimesteps(), optimization_stride, this->getLambda(), this->getAlpha(), getValueFunctionThreshold(),
        this->initial_state_d_, this->control_d_, this->control_noise_d_, this->control_std_dev_d_,
        this->trajectory_costs_d_, this->stream_, false);

    // Return the costs ->  nominal,  real costs
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));

    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(), trajectory_costs_nominal_d,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Launch the norm exponential kernels for the nominal costs and the real costs
    this->setBaseline(mppi_common::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS), 0);
    this->setBaseline(mppi_common::computeBaselineCost(trajectory_costs_nominal_.data(), NUM_ROLLOUTS), 1);

    // In this case this->gamma = 1 / lambda
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X, this->trajectory_costs_d_, 1.0 / this->getLambda(),
                                     this->getBaselineCost(0), this->stream_, false);
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X, trajectory_costs_nominal_d, 1.0 / this->getLambda(),
                                     this->getBaselineCost(1), this->stream_, false);

    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(), trajectory_costs_nominal_d,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Launch the weighted reduction kernel for the nominal costs and the real costs
    this->setNormalizer(mppi_common::computeNormalizer(this->trajectory_costs_.data(), NUM_ROLLOUTS), 0);
    this->setNormalizer(mppi_common::computeNormalizer(trajectory_costs_nominal_.data(), NUM_ROLLOUTS), 1);

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
  }
  // Smooth the control
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
  this->smoothControlTrajectoryHelper(nominal_control_trajectory_, nominal_control_history_);

  // Compute the nominal trajectory because we updated the nominal control!
  this->computeStateTrajectoryHelper(nominal_state_trajectory_, nominal_state_, nominal_control_trajectory_);

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

ROBUST_MPPI_TEMPLATE
float RobustMPPI::computeDF()
{
  return (this->getFeedbackPropagatedStateSeq().col(0) - this->getFeedbackPropagatedStateSeq().col(1)).norm() +
         (this->getTargetStateSeq().col(0) - this->getFeedbackPropagatedStateSeq().col(0)).norm();
}

ROBUST_MPPI_TEMPLATE
void RobustMPPI::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->getTotalSampledTrajectories();

  // control already copied in compute control, so run kernel
  mppi_common::launchStateAndCostTrajectoryKernel<DYN_T, COST_T, FEEDBACK_GPU, BDIM_X, BDIM_Y, 2>(
      this->model_->model_d_, this->cost_->cost_d_, this->fb_controller_->getDevicePointer(), this->sampled_noise_d_,
      this->initial_state_d_, this->sampled_outputs_d_, this->sampled_costs_d_, this->sampled_crash_status_d_,
      num_sampled_trajectories, this->getNumTimesteps(), this->getDt(), this->vis_stream_, getValueFunctionThreshold());

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
