#include <mppi/feedback_controllers/feedback.cuh>

// ===================== GPUFeedbackController ========================
template<class CLASS_T, class DYN_T, class FEEDBACK_STATE_T>
void GPUFeedbackController<CLASS_T, DYN_T, FEEDBACK_STATE_T>::copyToDevice(){
  if(GPUMemStatus_){
    HANDLE_ERROR(cudaMemcpyAsync(&feedback_d_->state_, &state_,
                                 sizeof(FEEDBACK_STATE_T), cudaMemcpyHostToDevice,
                                 stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }
}

template<class CLASS_T, class DYN_T, class FEEDBACK_STATE_T>
void GPUFeedbackController<CLASS_T, DYN_T, FEEDBACK_STATE_T>::freeCudaMem(){
  if(GPUMemStatus_) {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->deallocateCUDAMemory();
    cudaFree(feedback_d_);
    GPUMemStatus_ = false;
    feedback_d_ = nullptr;
  }
}

template<class CLASS_T, class DYN_T, class FEEDBACK_STATE_T>
void GPUFeedbackController<CLASS_T, DYN_T, FEEDBACK_STATE_T>::GPUSetup(){
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  if(!GPUMemStatus_) {
    feedback_d_ = Managed::GPUSetup(derived);
    derived->allocateCUDAMemory();
  } else {
    std::cout << "Feedback Controller GPU Memory already set" << std::endl;
  }
  derived->copyToDevice();
  // derived->paramsToDevice();
}

// ===================== FeedbackController ========================
