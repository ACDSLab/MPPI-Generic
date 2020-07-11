#include "robust_mppi_controller.cuh"
#include <exception>

#define RobustMPPI RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y, SAMPLES_PER_CONDITION_MULTIPLIER>


template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
RobustMPPI::RobustMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                     float lambda, float alpha,
                     float value_function_threshold,
                     const Eigen::Ref<const StateCostWeight>& Q,
                     const Eigen::Ref<const Hessian>& Qf,
                     const Eigen::Ref<const ControlCostWeight>& R,
                     const Eigen::Ref<const control_array>& control_std_dev,
                     int num_timesteps,
                     const Eigen::Ref<const control_trajectory>& init_control_traj,
                     int num_candidate_nominal_states,
                     int optimization_stride,
                     cudaStream_t stream) : Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
        model, cost, dt, max_iter, lambda, alpha, control_std_dev, num_timesteps, init_control_traj, stream),
        value_function_threshold_(value_function_threshold), optimization_stride_(optimization_stride),
        num_candidate_nominal_states_(num_candidate_nominal_states) {

  updateNumCandidates(num_candidate_nominal_states_);

  // Zero the control history
  this->control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();
  nominal_control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();

  // Initialize DDP
  this->initDDP(Q, Qf, R);

  // Initialize the nominal control trajectory
  nominal_control_trajectory_ = init_control_traj;

  // Resize the feedback gain vector to hold the raw data for the feedback gains
  feedback_gain_vector_.resize(MAX_TIMESTEPS*DYN_T::STATE_DIM*DYN_T::CONTROL_DIM);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
RobustMPPI::~RobustMPPIController() {
  deallocateNominalStateCandidateMemory();
  deallocateCUDAMemory();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::allocateCUDAMemory() {
  Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::allocateCUDAMemoryHelper(1);

  // We need to allocate memory for the feedback gains
  HANDLE_ERROR(cudaMalloc((void**)&feedback_gain_array_d_, sizeof(float)*DYN_T::STATE_DIM*DYN_T::CONTROL_DIM*this->num_timesteps_));
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::deallocateCUDAMemory() {
  HANDLE_ERROR(cudaFree(feedback_gain_array_d_));
}


template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::getInitNominalStateCandidates(
        const Eigen::Ref<const state_array>& nominal_x_k,
        const Eigen::Ref<const state_array>& nominal_x_kp1,
        const Eigen::Ref<const state_array>& real_x_kp1) {

  Eigen::MatrixXf points(DYN_T::STATE_DIM, 3);
  points << nominal_x_k, nominal_x_kp1 , real_x_kp1;
  auto candidates = points * line_search_weights_;
  for (int i = 0; i < num_candidate_nominal_states_; ++i) {
    candidate_nominal_states_[i] = candidates.col(i);
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::resetCandidateCudaMem() {
  deallocateNominalStateCandidateMemory();
  HANDLE_ERROR(cudaMalloc((void**)&importance_sampling_states_d_,
          sizeof(float)*DYN_T::STATE_DIM*num_candidate_nominal_states_));
  HANDLE_ERROR(cudaMalloc((void**)&importance_sampling_costs_d_,
                          sizeof(float)*num_candidate_nominal_states_*SAMPLES_PER_CONDITION));
  HANDLE_ERROR(cudaMalloc((void**)&importance_sampling_strides_d_,
                          sizeof(int)*num_candidate_nominal_states_));

  // Set flag so that the we know cudamemory is allocated
  importance_sampling_cuda_mem_init_ = true;
}



template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::deallocateNominalStateCandidateMemory() {
  if (importance_sampling_cuda_mem_init_) {
    HANDLE_ERROR(cudaFree(importance_sampling_states_d_));
    HANDLE_ERROR(cudaFree(importance_sampling_costs_d_));
    HANDLE_ERROR(cudaFree(importance_sampling_strides_d_));

    // Set flag so that we know cudamemory has been freed
    importance_sampling_cuda_mem_init_ = false;
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::copyNominalControlToDevice() {
  HANDLE_ERROR(cudaMemcpyAsync(this->control_d_, nominal_control_trajectory_.data(),
          sizeof(float)*nominal_control_trajectory_.size(), cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::updateNumCandidates(int new_num_candidates) {

  if ((new_num_candidates * SAMPLES_PER_CONDITION) > NUM_ROLLOUTS) {
    std::cerr << "ERROR: (number of candidates) * (SAMPLES_PER_CONDITION) cannot exceed NUM_ROLLOUTS\n";
    std::terminate();
  }

  // New number must be odd and greater than 3
  if (new_num_candidates < 3) {
    std::cerr << "ERROR: number of candidates must be greater or equal to 3\n";
    std::terminate();
  }
  if (new_num_candidates % 2 == 0) {
    std::cerr << "ERROR: number of candidates must be odd\n";
    std::terminate();
  }

  // Set the new value of the number of candidates
  num_candidate_nominal_states_ = new_num_candidates;

  // Resize the vector holding the candidate nominal states
  candidate_nominal_states_.resize(num_candidate_nominal_states_);

  // Resize the matrix holding the importance sampler strides
  importance_sampler_strides_.resize(1, num_candidate_nominal_states_);

  // Resize the trajectory costs matrix
  candidate_trajectory_costs_.resize(num_candidate_nominal_states_*SAMPLES_PER_CONDITION, 1);
  candidate_trajectory_costs_.setZero();

  // Resize the free energy costs matrix
  candidate_free_energy_.resize(num_candidate_nominal_states_, 1);
  candidate_free_energy_.setZero();

  // Deallocate and reallocate cuda memory
  resetCandidateCudaMem();

  // Recompute the line search weights based on the number of candidates
  computeLineSearchWeights();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::computeLineSearchWeights() {
  line_search_weights_.resize(3, num_candidate_nominal_states_);

  // For a given setup, this never changes.... why recompute every time?
  int num_candid_over_2 = num_candidate_nominal_states_/2;
  for (int i = 0; i < num_candid_over_2 + 1; i++){
    line_search_weights_(0, i) = 1 - i/float(num_candid_over_2);
    line_search_weights_(1, i) = i/float(num_candid_over_2);
    line_search_weights_(2, i) = 0.0;
  }
  for (int i = 1; i < num_candid_over_2 + 1; i++){
    line_search_weights_(0, num_candid_over_2 + i) = 0.0;
    line_search_weights_(1, num_candid_over_2 + i) = 1 - i/float(num_candid_over_2);
    line_search_weights_(2, num_candid_over_2 + i) = i/float(num_candid_over_2);
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::computeImportanceSamplerStride(int stride) {
  Eigen::MatrixXf stride_vec(1,3);
  stride_vec << 0, stride, stride;

  // Perform matrix multiplication, convert to array so that we can round the floats to the nearest
  // integer. Then cast the resultant float array to an int array. Then set equal to our int matrix.
  importance_sampler_strides_ = (stride_vec*line_search_weights_).array().round().template cast<int>();

}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
float RobustMPPI::computeCandidateBaseline() {
  float baseline = candidate_trajectory_costs_(0);
  for (int i = 0; i < SAMPLES_PER_CONDITION; i++){ // TODO What is the reasoning behind only using the first condition to get the baseline?
    if (candidate_trajectory_costs_(i) < baseline){
      baseline = candidate_trajectory_costs_(i);
    }
  }
  return baseline;
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::computeBestIndex() {
  candidate_free_energy_.setZero();
  float baseline = computeCandidateBaseline();
  for (int i = 0; i < num_candidate_nominal_states_; i++){
    for (int j = 0; j < SAMPLES_PER_CONDITION; j++){
      candidate_free_energy_(i) += expf(-1.0/this->lambda_*(candidate_trajectory_costs_(i*SAMPLES_PER_CONDITION + j) - baseline));
    }
    candidate_free_energy_(i) /= (1.0*SAMPLES_PER_CONDITION);
    candidate_free_energy_(i) = -this->lambda_ *logf(candidate_free_energy_(i)) + baseline;

    if (candidate_free_energy_(i) < value_function_threshold_){
      best_index_ = i;
    }
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::updateImportanceSamplingControl(const Eigen::Ref<const state_array> &state, int stride) {
  // (Controller Frequency)*(Optimization Time) corresponds to how many timesteps occurred in the last optimization
  real_stride_ = stride;

  computeNominalStateAndStride(state, stride); // Launches the init eval kernel

  // Save the nominal control history for the importance sampler
  this->saveControlHistoryHelper(nominal_stride_, nominal_control_trajectory_, nominal_control_history_);

  // Save the real control history for the optimal control
  this->saveControlHistoryHelper(real_stride_, this->control_, this->control_history_);

  // Slide the control sequence for the nominal control trajectory
  this->slideControlSequenceHelper(nominal_stride_, nominal_control_trajectory_);

  // Compute the nominal trajectory
  this->computeStateTrajectoryHelper(nominal_state_trajectory_, nominal_state_, nominal_control_trajectory_);

  // Compute the feedback gains and save them to an array
  computeNominalFeedbackGains(state);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::computeNominalStateAndStride(const Eigen::Ref<const state_array> &state, int stride) {
  if (!nominal_state_init_){
    nominal_state_ = state;
    nominal_state_init_ = true;
    nominal_stride_ = 0;
  } else {
    getInitNominalStateCandidates(nominal_state_trajectory_.col(0), nominal_state_trajectory_.col(1), state);
    computeImportanceSamplerStride(stride);

    // Send the nominal state candidates to the GPU
    HANDLE_ERROR( cudaMemcpyAsync(importance_sampling_states_d_, candidate_nominal_states_.data(),
                                  sizeof(float)*DYN_T::STATE_DIM*num_candidate_nominal_states_,
                                  cudaMemcpyHostToDevice, this->stream_));
    // Send the importance sampler strides to the GPU
    HANDLE_ERROR( cudaMemcpyAsync(importance_sampling_strides_d_, importance_sampler_strides_.data(),
                                  sizeof(int)*num_candidate_nominal_states_,
                                  cudaMemcpyHostToDevice, this->stream_));

    // Send the nominal control to the GPU
    copyNominalControlToDevice();

    // Generate noise for the samples
    curandGenerateNormal(this->gen_, this->control_noise_d_,
            SAMPLES_PER_CONDITION*num_candidate_nominal_states_*this->num_timesteps_*DYN_T::CONTROL_DIM, 0.0, 1.0);

    // Launch the init eval kernel
    rmppi_kernels::launchInitEvalKernel<DYN_T, COST_T, BDIM_X, BDIM_Y, SAMPLES_PER_CONDITION>(
            this->model_->model_d_, this->cost_->cost_d_, num_candidate_nominal_states_, this->num_timesteps_,
            this->lambda_, this->alpha_,
            optimization_stride_, this->dt_, importance_sampling_strides_d_, this->control_std_dev_d_,
            importance_sampling_states_d_, this->control_d_, this->control_noise_d_, importance_sampling_costs_d_,
            this->stream_);

    HANDLE_ERROR( cudaMemcpyAsync(candidate_trajectory_costs_.data(), importance_sampling_costs_d_,
            sizeof(float)*num_candidate_nominal_states_*SAMPLES_PER_CONDITION, cudaMemcpyDeviceToHost, this->stream_));
    cudaStreamSynchronize(this->stream_);

    // Compute the best nominal state candidate from the rollouts
    computeBestIndex();
    nominal_stride_ = importance_sampler_strides_(best_index_);
    nominal_state_ = candidate_nominal_states_[best_index_];
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::computeNominalFeedbackGains(const Eigen::Ref<const state_array> &state) {
  this->run_cost_->setTargets(nominal_state_trajectory_.data(), nominal_control_trajectory_.data(),
                        this->num_timesteps_);

  this->terminal_cost_->xf = this->run_cost_->traj_target_x_.col(this->num_timesteps_ - 1);
  this->result_ = this->ddp_solver_->run(state, nominal_control_trajectory_,
                             *this->ddp_model_, *this->run_cost_, *this->terminal_cost_,
                             this->control_min_, this->control_max_);

  // Copy the feedback gains into the std::vector (this is useful for easily copying into GPU memory
  // Copy Feedback Gains into an array
  for (size_t i = 0; i < this->result_.feedback_gain.size(); i++) {
    int i_index = i * DYN_T::STATE_DIM * DYN_T::CONTROL_DIM;
    for (size_t j = 0; j < DYN_T::CONTROL_DIM * DYN_T::STATE_DIM; j++) {
      feedback_gain_vector_[i_index + j] = this->result_.feedback_gain[i].data()[j];
    }
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y, int SAMPLES_PER_CONDITION_MULTIPLIER>
void RobustMPPI::computeControl(const Eigen::Ref<const state_array> &state, int optimization_stride) {
  // Handy dandy pointers to nominal data
  float * trajectory_costs_nominal_d = this->trajectory_costs_d_ + NUM_ROLLOUTS;
  float * initial_state_nominal_d = this->initial_state_d_ + DYN_T::STATE_DIM;
  float * control_noise_nominal_d = this->control_noise_d_ + NUM_ROLLOUTS * this->num_timesteps_ * DYN_T::CONTROL_DIM;
  float * control_nominal_d = this->control_d_ + this->num_timesteps_ * DYN_T::CONTROL_DIM;

  // Transfer the feedback gains to the GPU
  HANDLE_ERROR(cudaMemcpyAsync(feedback_gain_array_d_, feedback_gain_vector_.data(),
                               sizeof(float)*this->num_timesteps_*DYN_T::STATE_DIM*DYN_T::CONTROL_DIM,
                               cudaMemcpyHostToDevice, this->stream_));
  // Transfer the real initial state to the GPU
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), sizeof(float)*DYN_T::STATE_DIM,
          cudaMemcpyHostToDevice, this->stream_));
  // Transfer the nominal state to the GPU: recall that the device GPU has the augmented state [real state, nominal state]
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_nominal_d, nominal_state_.data(), sizeof(float)*DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, this->stream_));

  for (int opt_iter = 0; opt_iter < this->num_iters_; opt_iter++) {
    // Copy the importance sampling control to the system
    HANDLE_ERROR( cudaMemcpyAsync(this->control_d_, nominal_control_trajectory_.data(),
            sizeof(float)*DYN_T::CONTROL_DIM*this->num_timesteps_, cudaMemcpyHostToDevice, this->stream_));

    // Generate a the control perturbations for exploration
    curandGenerateNormal(this->gen_, this->control_noise_d_, DYN_T::CONTROL_DIM*this->num_timesteps_*NUM_ROLLOUTS, 0.0, 1.0);

    //Make a second copy of the random deviations
    HANDLE_ERROR( cudaMemcpyAsync(control_noise_nominal_d, this->control_noise_d_,
            DYN_T::CONTROL_DIM*this->num_timesteps_*NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToDevice, this->stream_));

    // Launch the new rollout kernel
    rmppi_kernels::launchRMPPIRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BLOCKSIZE_X,
            BLOCKSIZE_Y, 2>(this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_, optimization_stride,
                            this->lambda_, this->alpha_, value_function_threshold_, this->initial_state_d_, this->control_d_,
                            this->control_noise_d_, feedback_gain_array_d_, this->control_std_dev_d_,
                            this->trajectory_costs_d_, this->stream_);

    // Return the costs ->  nominal,  real costs
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(),
                                 this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float),
                                 cudaMemcpyDeviceToHost, this->stream_));

    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(),
                                 trajectory_costs_nominal_d,
                                 NUM_ROLLOUTS * sizeof(float),
                                 cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Launch the norm exponential kernels for the nominal costs and the real costs
    this->baseline_ = mppi_common::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS);
    baseline_nominal_ = mppi_common::computeBaselineCost(trajectory_costs_nominal_.data(), NUM_ROLLOUTS);

    // In this case this->gamma = 1 / lambda
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
                                     this->trajectory_costs_d_, 1.0 / this->lambda_, this->baseline_, this->stream_);
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
                                     trajectory_costs_nominal_d, 1.0 / this->lambda_, baseline_nominal_, this->stream_);

    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(), trajectory_costs_nominal_d,
                                 NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Launch the weighted reduction kernel for the nominal costs and the real costs
    this->normalizer_ = mppi_common::computeNormalizer(this->trajectory_costs_.data(), NUM_ROLLOUTS);
    normalizer_nominal_ = mppi_common::computeNormalizer(trajectory_costs_nominal_.data(), NUM_ROLLOUTS);

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
  }
  // Smooth the control
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
  this->smoothControlTrajectoryHelper(nominal_control_trajectory_, nominal_control_history_);
}
