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
void DeviceDDPImpl<GPU_FB_T, DYN_T>::copyToDevice() {
  int fb_size = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * num_timesteps_;
  HANDLE_ERROR(cudaMemcpyAsync(this->feedback_d_->fb_gains_,
                               fb_gains_,
                               sizeof(float) * fb_size,
                               cudaMemcpyHostToDevice,
                               this->stream_));
  HANDLE_ERROR( cudaStreamSynchronize(this->stream_) );
}

void DDPFeedback::initTrackingController() {
  util::DefaultLogger logger;
  bool verbose = false;
  ddp_model_  = std::make_shared<ModelWrapperDDP<DYN_T>>(model_);
  ddp_solver_ = std::make_shared< DDP<ModelWrapperDDP<DYN_T>>>(dt_,
                                                               num_timesteps_, 1, &logger, verbose);
  Q_ = q_mat;
  Qf_ = q_f_mat;
  R_ = r_mat;

  for (int i = 0; i < DYN_T::CONTROL_DIM; i++) {
    control_min_(i) = model_->control_rngs_[i].x;
    control_max_(i) = model_->control_rngs_[i].y;
  }

  run_cost_ = std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(Q_,
                                                                        R_, num_timesteps_);
  terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(Qf_);
}

