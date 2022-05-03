#include <mppi/controllers/controller.cuh>

#define CONTROLLER Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::deallocateCUDAMemory()
{
  cudaFree(control_d_);
  cudaFree(state_d_);
  cudaFree(trajectory_costs_d_);
  cudaFree(control_std_dev_d_);
  cudaFree(control_noise_d_);
  if (sampled_states_CUDA_mem_init_)
  {
    cudaFree(sampled_states_d_);
    cudaFree(sampled_noise_d_);
    cudaFree(sampled_costs_d_);
    sampled_states_CUDA_mem_init_ = false;
  }
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::copyControlStdDevToDevice()
{
  HANDLE_ERROR(cudaMemcpyAsync(control_std_dev_d_, control_std_dev_.data(), sizeof(float) * control_std_dev_.size(),
                               cudaMemcpyHostToDevice, stream_));
  cudaStreamSynchronize(stream_);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::copyNominalControlToDevice()
{
  HANDLE_ERROR(
      cudaMemcpyAsync(control_d_, control_.data(), sizeof(float) * control_.size(), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR(cudaStreamSynchronize(stream_));
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::copySampledControlFromDevice(bool synchronize)
{
  // if mem is not inited don't use it
  if (!sampled_states_CUDA_mem_init_)
  {
    return;
  }

  int num_sampled_trajectories = perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  std::vector<int> samples(num_sampled_trajectories);
  if (this->perc_sampled_control_trajectories_ > 0.98)
  {
    // if above threshold just do everything
    std::iota(samples.begin(), samples.end(), 0);
  }
  else
  {
    // Create sample list without replacement
    // removes the top 2% since top 1% are complete noise
    samples = mppi_math::sample_without_replacement(num_sampled_trajectories, NUM_ROLLOUTS * 0.98);
  }

  // this explicitly adds the optimized control sequence
  HANDLE_ERROR(cudaMemcpyAsync(this->sampled_noise_d_, this->control_d_,
                               sizeof(float) * this->num_timesteps_ * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                               this->vis_stream_));

  for (int i = 1; i < num_sampled_trajectories; i++)
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_noise_d_ + i * this->num_timesteps_ * DYN_T::CONTROL_DIM,
                                 this->control_noise_d_ + samples[i] * this->num_timesteps_ * DYN_T::CONTROL_DIM,
                                 sizeof(float) * this->num_timesteps_ * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
  }
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
std::pair<int, float> CONTROLLER::findMinIndexAndValue(std::vector<int>& temp_list)
{
  if (temp_list.size() == 0)
  {
    return std::make_pair(0, 0.0);
  }
  int min_sample_index = 0;
  float min_sample_value = this->trajectory_costs_[temp_list[min_sample_index]];

  for (int index = 1; index < temp_list.size(); index++)
  {
    if (this->trajectory_costs_[temp_list[index]] < min_sample_value)
    {
      min_sample_value = this->trajectory_costs_[temp_list[index]];
      min_sample_index = index;
    }
  }
  return std::make_pair(min_sample_index, min_sample_value);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::copyTopControlFromDevice(bool synchronize)
{
  // if mem is not inited don't use it
  if (!sampled_states_CUDA_mem_init_ || num_top_control_trajectories_ <= 0)
  {
    return;
  }

  // Important note: Highest weighted trajectories are the ones with the lowest cost
  int start_top_control_traj_index = perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  std::vector<int> samples(num_top_control_trajectories_);
  // Start by filling in the top samples list with the first n in the trajectory
  for (int i = 0; i < num_top_control_trajectories_; i++)
  {
    samples[i] = i;
  }

  // Calculate min weight in the current top samples list
  int min_sample_index = 0;
  float min_sample_value = 0;
  std::tie(min_sample_index, min_sample_value) = findMinIndexAndValue(samples);

  // find top n samples by removing the smallest weights from the list
  for (int i = num_top_control_trajectories_; i < NUM_ROLLOUTS; i++)
  {
    if (trajectory_costs_[i] > min_sample_value)
    {  // Remove the smallest weight in the current list and add the new index
      samples[min_sample_index] = i;
      // recalculate min weight in the current list
      std::tie(min_sample_index, min_sample_value) = findMinIndexAndValue(samples);
    }
  }

  // Copy top n samples to this->sampled_noise_d_ after the randomly sampled trajectories
  top_n_costs_.resize(num_top_control_trajectories_);
  for (int i = 0; i < num_top_control_trajectories_; i++)
  {
    top_n_costs_[i] = trajectory_costs_[samples[i]] / normalizer_;
    HANDLE_ERROR(cudaMemcpyAsync(
        this->sampled_noise_d_ + (start_top_control_traj_index + i) * this->num_timesteps_ * DYN_T::CONTROL_DIM,
        this->control_noise_d_ + samples[i] * this->num_timesteps_ * DYN_T::CONTROL_DIM,
        sizeof(float) * this->num_timesteps_ * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice, this->vis_stream_));
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
  }
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::setCUDAStream(cudaStream_t stream)
{
  stream_ = stream;
  model_->bindToStream(stream);
  cost_->bindToStream(stream);
  fb_controller_->bindToStream(stream);
  curandSetStream(gen_, stream);  // requires the generator to be created!
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::createAndSeedCUDARandomNumberGen()
{
  // Seed the PseudoRandomGenerator with the CPU time.
  curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  curandSetPseudoRandomGeneratorSeed(gen_, seed);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void CONTROLLER::allocateCUDAMemoryHelper(int nominal_size, bool allocate_double_noise)
{
  if (nominal_size < 0)
  {
    nominal_size = 1;
    std::cerr << "nominal size cannot be below 0 when allocateCudaMemoryHelper is called" << std::endl;
    std::exit(-1);
  }
  else
  {
    // increment by 1 since actual is not included
    ++nominal_size;
  }
  HANDLE_ERROR(cudaMalloc((void**)&initial_state_d_, sizeof(float) * DYN_T::STATE_DIM * nominal_size));
  HANDLE_ERROR(cudaMalloc((void**)&control_d_, sizeof(float) * DYN_T::CONTROL_DIM * MAX_TIMESTEPS * nominal_size));
  HANDLE_ERROR(cudaMalloc((void**)&state_d_, sizeof(float) * DYN_T::STATE_DIM * MAX_TIMESTEPS * nominal_size));
  HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d_, sizeof(float) * NUM_ROLLOUTS * nominal_size));
  HANDLE_ERROR(cudaMalloc((void**)&control_std_dev_d_, sizeof(float) * DYN_T::CONTROL_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&control_noise_d_, sizeof(float) * DYN_T::CONTROL_DIM * MAX_TIMESTEPS * NUM_ROLLOUTS *
                                                         (allocate_double_noise ? nominal_size : 1)));
}

#undef CONTROLLER
