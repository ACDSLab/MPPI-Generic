#include <mppi/feedback_controllers/DDP/ddp.cuh>

/**
 * Methods for DDPFeedbackState
 */

template <class DYN_T>
DDPFeedbackState<DYN_T>::DDPFeedbackState(int num_timesteps, cudaStream_t stream)
{
  setNumTimesteps(num_timesteps);
  setCUDAStream(stream);
}

template <class DYN_T>
DDPFeedbackState<DYN_T>::DDPFeedbackState(const DDPFeedbackState<DYN_T>& other)
{
  setCUDAStream(other.stream_);
  setNumTimesteps(other.getNumTimesteps());  // this creates the CPU memory fb_gain_traj_
  if (other.fb_gain_traj_d_)
  {
    allocateCUDAMemory();
    HANDLE_ERROR(cudaMemcpyAsync(fb_gain_traj_d_, other.fb_gain_traj_d_, sizeof(float) * size(),
                                 cudaMemcpyDeviceToDevice, stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }
  for (int i = 0; i < size(); i++)
  {
    fb_gain_traj_[i] = other.fb_gain_traj_[i];
  }
}

// This method is not a member method but it can access member variables as it is friended
template <class DYN_T>
inline void swap(DDPFeedbackState<DYN_T>& first, DDPFeedbackState<DYN_T>& second)
{
  using std::swap;  // Declare like this for ADL purposes
  swap(first.num_timesteps_, second.num_timesteps_);
  swap(first.stream_, second.stream_);
  swap(first.fb_gain_traj_, second.fb_gain_traj_);
  swap(first.fb_gain_traj_d_, second.fb_gain_traj_d_);
}

// Let the compiler create a new DDPFeedbackState by explicitly using pass by value
template <class DYN_T>
DDPFeedbackState<DYN_T>& DDPFeedbackState<DYN_T>::operator=(DDPFeedbackState<DYN_T> other)
{
  swap(*this, other);
  return *this;
}

template <class DYN_T>
DDPFeedbackState<DYN_T>::~DDPFeedbackState()
{
  if (fb_gain_traj_ != nullptr)
  {
    delete[] fb_gain_traj_;
    fb_gain_traj_ = nullptr;
  }
  deallocateCUDAMemory();
}

template <class DYN_T>
void DDPFeedbackState<DYN_T>::setNumTimesteps(const int num_timesteps)
{
  const int prev_timesteps = this->num_timesteps_;
  const bool larger_array_needed = num_timesteps > prev_timesteps;
  this->num_timesteps_ = num_timesteps;
  if (larger_array_needed)
  {
    allocateCUDAMemory();

    // float* fb_gain_traj_new = new float[size()](); // initialize to zero
    if (fb_gain_traj_ != nullptr)
    {
      // for (int i = 0; i < prev_timesteps i++)
      // {
      //   fb_gain_traj_new[i] = fb_gain_traj_[i];
      // }
      delete[] fb_gain_traj_;
    }
    fb_gain_traj_ = new float[size()]();  // Initialize to zero
    // fb_gain_traj_ = fb_gain_traj_new;
  }
}

template <class DYN_T>
void DDPFeedbackState<DYN_T>::setCUDAStream(const cudaStream_t stream)
{
  stream_ = stream;
}

template <class DYN_T>
__host__ __device__ int DDPFeedbackState<DYN_T>::getNumTimesteps() const
{
  return num_timesteps_;
}

template <class DYN_T>
__host__ __device__ float* DDPFeedbackState<DYN_T>::getFeedbackGainPtr() const
{
#ifdef __CUDA_ARCH__
  return fb_gain_traj_d_;
#else
  return fb_gain_traj_;
#endif
}

template <class DYN_T>
__host__ __device__ const float* DDPFeedbackState<DYN_T>::getConstFeedbackGainPtr() const
{
#ifdef __CUDA_ARCH__
  return fb_gain_traj_d_;
#else
  return fb_gain_traj_;
#endif
}

template <class DYN_T>
__host__ __device__ std::size_t DDPFeedbackState<DYN_T>::size() const
{
  return DYN_T::CONTROL_DIM * DYN_T::STATE_DIM * getNumTimesteps();
}

template <class DYN_T>
__host__ void DDPFeedbackState<DYN_T>::allocateCUDAMemory()
{
  if (fb_gain_traj_d_)
  {
    deallocateCUDAMemory();
  }
  HANDLE_ERROR(cudaMalloc((void**)&fb_gain_traj_d_, sizeof(float) * size()));
}

template <class DYN_T>
__host__ void DDPFeedbackState<DYN_T>::deallocateCUDAMemory()
{
  if (fb_gain_traj_d_)
  {
    HANDLE_ERROR(cudaFree(fb_gain_traj_d_));
    fb_gain_traj_d_ = nullptr;
  }
}

template <class DYN_T>
__host__ void DDPFeedbackState<DYN_T>::copyToDevice(bool synchronize)
{
  HANDLE_ERROR(
      cudaMemcpyAsync(fb_gain_traj_d_, fb_gain_traj_, sizeof(float) * size(), cudaMemcpyHostToDevice, stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }
}

template <class DYN_T>
bool DDPFeedbackState<DYN_T>::isEqual(const DDPFeedbackState<DYN_T>& other) const
{
  if (this->getNumTimesteps() != other.getNumTimesteps())
  {
    return false;
  }
  for (int i = 0; i < this->size(); i++)
  {
    if (this->fb_gain_traj_[i] != other.fb_gain_traj_[i])
    {
      return false;
    }
  }
  return true;
}

/**
 * GPU Class for DDP Methods
 */
template <class GPU_FB_T, class DYN_T>
DeviceDDPImpl<GPU_FB_T, DYN_T>::DeviceDDPImpl(int num_timesteps, cudaStream_t stream) : PARENT_CLASS(stream)
{
  this->setNumTimesteps(num_timesteps);
  this->state_.setCUDAStream(stream);
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::allocateCUDAMemory()
{
  // this->state_.allocateCUDAMemory is not needed as it is done within setNumTimesteps()
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::copyToDevice(bool synchronize)
{
  this->state_.copyToDevice(false);         // Copy gains to GPU first
  PARENT_CLASS::copyToDevice(synchronize);  // Copy num_timesteps and fb_gain_ptrs to GPU
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::deallocateCUDAMemory()
{
  this->state_.deallocateCUDAMemory();
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::setNumTimesteps(const int num_timesteps)
{
  this->state_.setNumTimesteps(num_timesteps);
}

template <class GPU_FB_T, class DYN_T>
__host__ __device__ int DeviceDDPImpl<GPU_FB_T, DYN_T>::getNumTimesteps() const
{
  return this->state_.getNumTimesteps();
}

template <class GPU_FB_T, class DYN_T>
__device__ void DeviceDDPImpl<GPU_FB_T, DYN_T>::k(const float* __restrict__ x_act, const float* __restrict__ x_goal,
                                                  const int t, float* __restrict__ theta,
                                                  float* __restrict__ control_output)
{
  const float* fb_gain_t = &(this->state_.getConstFeedbackGainPtr()[DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t]);
  float e = 0;
  for (int i = 0; i < DYN_T::STATE_DIM; i++)
  {
    e = x_act[i] - x_goal[i];
    if (DYN_T::CONTROL_DIM % 4 == 0)
    {  // load 4 floats in at a time to save on global memory reads
      const float4* fb_gain_t4 = reinterpret_cast<const float4*>(&fb_gain_t[i * DYN_T::CONTROL_DIM]);
      for (int j = 0; j < DYN_T::CONTROL_DIM / 4; j++)
      {
        reinterpret_cast<float4*>(control_output)[j] = fb_gain_t4[j] * e;
      }
    }
    else if (DYN_T::CONTROL_DIM % 2 == 0)
    {  // load 2 floats in at a time to save on global memory reads
      const float2* fb_gain_t2 = reinterpret_cast<const float2*>(&fb_gain_t[i * DYN_T::CONTROL_DIM]);
      for (int j = 0; j < DYN_T::CONTROL_DIM / 2; j++)
      {
        reinterpret_cast<float2*>(control_output)[j] = fb_gain_t2[j] * e;
      }
    }
    else
    {
      for (int j = 0; j < DYN_T::CONTROL_DIM; j++)
      {
        control_output[j] += fb_gain_t[i * DYN_T::CONTROL_DIM + j] * e;
      }
    }
  }
}

/**
 * CPU Class for DDP Methods
 */
template <class DYN_T>
DDPFeedback<DYN_T>::DDPFeedback(DYN_T* model, float dt, int num_timesteps, cudaStream_t stream)
{
  model_ = model;
  this->setDt(dt);
  this->setNumTimesteps(num_timesteps);
  this->gpu_controller_->freeCudaMem();  // Remove allocated CUDA mem from default constructor
  this->gpu_controller_ = std::make_shared<DeviceDDP<DYN_T>>(this->num_timesteps_, stream);
}

template <class DYN_T>
void DDPFeedback<DYN_T>::initTrackingController()
{
  util::DefaultLogger logger;
  bool verbose = false;
  ddp_model_ = std::make_shared<ModelWrapperDDP<DYN_T>>(model_);
  ddp_solver_ = std::make_shared<DDP<ModelWrapperDDP<DYN_T>>>(this->dt_, this->num_timesteps_,
                                                              this->params_.num_iterations, &logger, verbose);

  result_ = OptimizerResult<ModelWrapperDDP<DYN_T>>();
  result_.feedback_gain = feedback_gain_trajectory(this->num_timesteps_);
  for (int i = 0; i < this->num_timesteps_; i++)
  {
    result_.feedback_gain[i] = DYN_T::feedback_matrix::Zero();
  }

  run_cost_ =
      std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(this->params_.Q, this->params_.R, this->num_timesteps_);
  terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(this->params_.Q_f);
}

template <class DYN_T>
void DDPFeedback<DYN_T>::setParams(const DDPParams<DYN_T>& params)
{
  this->params_ = params;
  run_cost_ =
      std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(this->params_.Q, this->params_.R, this->num_timesteps_);
  terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(this->params_.Q_f);
}

template <class DYN_T>
void DDPFeedback<DYN_T>::setNumTimesteps(const int num_timesteps)
{
  PARENT_CLASS::setNumTimesteps(num_timesteps);
  this->gpu_controller_->setNumTimesteps(num_timesteps);
}

template <class DYN_T>
void DDPFeedback<DYN_T>::computeFeedback(const Eigen::Ref<const state_array>& init_state,
                                         const Eigen::Ref<const state_trajectory>& goal_traj,
                                         const Eigen::Ref<const control_trajectory>& control_traj)
{
  run_cost_->setTargets(goal_traj.data(), control_traj.data(), this->num_timesteps_);

  terminal_cost_->xf = run_cost_->traj_target_x_.col(this->num_timesteps_ - 1);

  // update control ranges
  for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
  {
    control_min_(i) = model_->control_rngs_[i].x;
    control_max_(i) = model_->control_rngs_[i].y;
  }

  result_ =
      ddp_solver_->run(init_state, control_traj, *ddp_model_, *run_cost_, *terminal_cost_, control_min_, control_max_);

  // Copy Feedback Gains into Feedback State
  for (size_t i = 0; i < result_.feedback_gain.size(); i++)
  {
    int i_index = i * DYN_T::STATE_DIM * DYN_T::CONTROL_DIM;
    for (size_t j = 0; j < DYN_T::CONTROL_DIM * DYN_T::STATE_DIM; j++)
    {
      this->getFeedbackStatePointer()->getFeedbackGainPtr()[i_index + j] = result_.feedback_gain[i].data()[j];
    }
  }
  // Actually put new feedback gain trajectory onto the GPU
  // this->gpu_controller_->copyToDevice();
}
