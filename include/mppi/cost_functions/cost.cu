#include <mppi/cost_functions/cost.cuh>

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Cost<CLASS_T, PARAMS_T, S_DIM, C_DIM>::paramsToDevice() {
  if(GPUMemStatus_){
    HANDLE_ERROR(cudaMemcpyAsync(&cost_d_->params_, &params_,
                                 sizeof(PARAMS_T), cudaMemcpyHostToDevice,
                                 stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }
}


template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Cost<CLASS_T, PARAMS_T, S_DIM, C_DIM>::freeCudaMem() {
  if(GPUMemStatus_) {
    cudaFree(cost_d_);
    GPUMemStatus_ = false;
    cost_d_ = nullptr;
  }
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Cost<CLASS_T, PARAMS_T, S_DIM, C_DIM>::GPUSetup() {
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  std::cout << typeid(CLASS_T).name() << std::endl;
  if (!GPUMemStatus_) {
    cost_d_ = Managed::GPUSetup<CLASS_T>(derived);
    std::cout << typeid(cost_d_).name() << std::endl;
  } else {
    std::cout << "GPU Memory already set" << std::endl; //TODO should this be an exception?
  }
  derived->paramsToDevice();
}
