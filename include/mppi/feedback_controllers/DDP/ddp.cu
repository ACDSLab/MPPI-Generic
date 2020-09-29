#include <mppi/feedback_controllers/DDP/ddp.cuh>

template <class CLASS_T, class DYN_T>
DeviceDDPImpl::DeviceDDPImpl(int num_timesteps, cudaStream_t stream = 0) :
  num_timesteps_(num_timesteps),
  GPUFeedbackController<CLASS_T, DYN_T>(stream) {}

template <class CLASS_T, class DYN_T>
DeviceDDPImpl::allocateCUDAMemory() {
  int fb_size = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * num_timesteps_;
  cudaMalloc((void**)&feedback_d_->fb_gains_, fb_size * sizeof(float));
}

template <class CLASS_T, class DYN_T>
DeviceDDPImpl::deallocateCUDAMemory() {
  if (feedback_d_->fb_gains_ != nullptr) {
    cudaFree(feedback_d_->fb_gains_);
    feedback_d_->fb_gains_ = nullptr;
  }
}

template <class CLASS_T, class DYN_T>
DeviceDDPImpl::copyToDevice() {
  int fb_size = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * num_timesteps_;
  HANDLE_ERROR(cudaMemcpyAsync(feedback_d_->fb_gains_,
                               fb_gains_,
                               sizeof(float) * fb_size,
                               cudaMemcpyHostToDevice
                               this->stream_));
  HANDLE_ERROR( cudaStreamSynchronize(this->stream_) );
}