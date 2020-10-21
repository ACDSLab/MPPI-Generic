#include <mppi/feedback_controllers/DDP/ddp.cuh>

template <class GPU_FB_T, class DYN_T>
DeviceDDPImpl<GPU_FB_T, DYN_T>::DeviceDDPImpl(int num_timesteps, cudaStream_t stream) :
  num_timesteps_(num_timesteps), GPUFeedbackController<GPU_FB_T, DYN_T>(stream) {

}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::allocateCUDAMemory() {
  int fb_size = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * num_timesteps_;
  cudaMalloc((void**)&this->feedback_d_->fb_gains_, fb_size * sizeof(float));
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::deallocateCUDAMemory() {
  if (this->feedback_d_->fb_gains_ != nullptr) {
    cudaFree(this->feedback_d_->fb_gains_);
    this->feedback_d_->fb_gains_ = nullptr;
  }
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::k(const float * x_act, const float * x_goal,
                                       const float t, float * theta,
                                       float* control_output) {
  float * fb_gain_t = &fb_gains_[DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t];
  float e = 0;
  for (int i = 0; i < DYN_T::STATE_DIM; i++) {
    e = x_act[i] - x_goal[i];
    for(int j = 0; j < DYN_CONTROL_DIM; j++) {
      control_output[j] += fb_gain_t[i * DYN_T::CONTROL_DIM + j] * e;
    }
  }
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::copyToDevice() {
  int fb_size = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * num_timesteps_;
  HANDLE_ERROR(cudaMemcpyAsync(this->feedback_d_->fb_gains_,
                               fb_gains_,
                               sizeof(float) * fb_size,
                               cudaMemcpyHostToDevice,
                               this->stream_));
  HANDLE_ERROR( cudaStreamSynchronize(this->stream_) );
}

template <class DYN_T, int NUM_TIMESTEPS>
void DDPFeedback<DYN_T, NUM_TIMESTEPS>::DDPFeedback(cudaStream_t stream) {
  this->gpu_controller_ = std::make_shared<DeviceDDP<DYN_T>>(NUM_TIMESTEPS, stream);
  this->gpu_controller_->GPUSetup();
}

template <class DYN_T, int NUM_TIMESTEPS>
void DDPFeedback<DYN_T, NUM_TIMESTEPS>::initTrackingController() {
  util::DefaultLogger logger;
  bool verbose = false;
  ddp_model_  = std::make_shared<ModelWrapperDDP<DYN_T>>(model_);
  ddp_solver_ = std::make_shared< DDP<ModelWrapperDDP<DYN_T>>>(dt_,
                                                               num_timesteps_,
                                                               1,
                                                               &logger,
                                                               verbose);
  // TODO: Figure out where q_mat, q_f_mat, and r_mat will be coming from
  this->params_.Q = q_mat;
  this->params_.Q_f = q_f_mat;
  this->params_.R = r_mat;

  // TODO: Are these control ranges needed?
  for (int i = 0; i < DYN_T::CONTROL_DIM; i++) {
    control_min_(i) = model_->control_rngs_[i].x;
    control_max_(i) = model_->control_rngs_[i].y;
  }

  run_cost_ = std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(this->params_.Q,
                                                                        this->params_.R,
                                                                        NUM_TIMESTEPS);
  terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(this->params_.Q_f);
}

template <class DYN_T, int NUM_TIMESTEPS>
control_array DDPFeedback<DYN_T, NUM_TIMESTEPS>::k(const Eigen::Ref<state_array>& x_act,
                                                   const Eigen::Ref<state_array>& x_goal,
                                                   float t) {
  // print
  control_array u_output = fb_gain_traj_[t] * (x_act - x_goal);
  return u_output;
}

template <class DYN_T, int NUM_TIMESTEPS>
control_array DDPFeedback<DYN_T, NUM_TIMESTEPS>::computeFeedbackGains(
    const Eigen::Ref<const state_array>& init_state,
    const Eigen::Ref<const state_trajectory>& goal_traj,
    const Eigen::Ref<const control_trajectory>& control_traj) {
  // TODO: Fill in later
}