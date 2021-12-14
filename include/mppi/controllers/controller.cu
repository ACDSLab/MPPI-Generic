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
void CONTROLLER::copySampledControlFromDevice()
{
  int num_sampled_trajectories = perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  // Create sample list without replacement
  std::vector<int> samples = mppi_math::sample_without_replacement(num_sampled_trajectories, NUM_ROLLOUTS);
  // Ensure that sampled_controls_ has enough space for the trajectories

  for (int i = 0; i < num_sampled_trajectories; i++)
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_noise_d_ + i * this->num_timesteps_ * DYN_T::CONTROL_DIM,
                                 this->control_noise_d_ + samples[i] * this->num_timesteps_ * DYN_T::CONTROL_DIM,
                                 sizeof(float) * this->num_timesteps_ * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->stream_));
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
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
