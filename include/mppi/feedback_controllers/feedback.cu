#include <mppi/feedback_controllers/feedback.cuh>

template<class CLASS_T, class DYN_T>
void GPUFeedbackController<CLASS_T, DYN_T>::freeCudaMem(){
  if(GPUMemStatus_) {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->deallocateCudaMemory();
    cudaFree(feedback_d_);
    GPUMemStatus_ = false;
    feedback_d_ = nullptr;
  }
}

template<class CLASS_T, class DYN_T>
void GPUFeedbackController<CLASS_T, DYN_T>::GPUSetup(){
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  if(!GPUMemStatus_) {
    feedback_d_ = Managed::GPUSetup(derived);
    derived->allocateCudaMemory();
  } else {
    std::cout << "GPU Memory already set" << std::endl;
  }
  derived->copyToDevice();
}