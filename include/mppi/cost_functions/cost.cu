#include <mppi/cost_functions/cost.cuh>

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Cost<CLASS_T, PARAMS_T, S_DIM, C_DIM>::paramsToDevice() {
  if(this->GPUMemStatus_){
    HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->params_, &this->params_,
                                 sizeof(PARAMS_T), cudaMemcpyHostToDevice,
                                 this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}


template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Cost<CLASS_T, PARAMS_T, S_DIM, C_DIM>::freeCudaMem() {
  if(GPUMemStatus_) {
    cudaFree(cost_d_);
    this->GPUMemStatus_ = false;
    cost_d_ = nullptr;
  }
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Cost<CLASS_T, PARAMS_T, S_DIM, C_DIM>::GPUSetup() {
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  if (!GPUMemStatus_) {
    this->cost_d_ = Managed::GPUSetup(derived);
  } else {
    std::cout << "GPU Memory already set" << std::endl; //TODO should this be an exception?
  }
  derived->paramsToDevice();
}