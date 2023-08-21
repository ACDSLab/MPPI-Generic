#include <mppi/cost_functions/cost.cuh>

template <class CLASS_T, class PARAMS_T, class DYN_PARAMS_T>
void Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>::paramsToDevice()
{
  if (GPUMemStatus_)
  {
    HANDLE_ERROR(cudaMemcpyAsync(&cost_d_->params_, &params_, sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }
}

template <class CLASS_T, class PARAMS_T, class DYN_PARAMS_T>
void Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>::freeCudaMem()
{
  if (GPUMemStatus_)
  {
    cudaFree(cost_d_);
    GPUMemStatus_ = false;
    cost_d_ = nullptr;
  }
}

template <class CLASS_T, class PARAMS_T, class DYN_PARAMS_T>
void Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>::GPUSetup()
{
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  if (!GPUMemStatus_)
  {
    cost_d_ = Managed::GPUSetup<CLASS_T>(derived);
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;  // TODO should this be an exception?
  }
  derived->paramsToDevice();
}

template <class CLASS_T, class PARAMS_T, class DYN_PARAMS_T>
__device__ float Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>::computeRunningCost(float* y, float* u, int timestep,
                                                                           float* theta_c, int* crash)
{
  if (threadIdx.y == 0)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    return derived->computeStateCost(y, timestep, theta_c, crash) +
           derived->computeControlCost(u, timestep, theta_c, crash);
  }
  else
  {
    return 0.0f;
  }
}
