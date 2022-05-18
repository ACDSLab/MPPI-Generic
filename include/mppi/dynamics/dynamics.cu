#include <mppi/dynamics/dynamics.cuh>

template <class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::paramsToDevice(bool synchronize)
{
  if (GPUMemStatus_)
  {
    HANDLE_ERROR(cudaMemcpyAsync(&model_d_->params_, &params_, sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));

    HANDLE_ERROR(cudaMemcpyAsync(&model_d_->control_rngs_, &control_rngs_, C_DIM * sizeof(float2),
                                 cudaMemcpyHostToDevice, stream_));
    if (synchronize)
    {
      HANDLE_ERROR(cudaStreamSynchronize(stream_));
    }
  }
}

template <class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::setControlRanges(std::array<float2, C_DIM>& control_rngs,
                                                                 bool synchronize)
{
  for (int i = 0; i < C_DIM; i++)
  {
    control_rngs_[i].x = control_rngs[i].x;
    control_rngs_[i].y = control_rngs[i].y;
  }
  if (GPUMemStatus_)
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->model_d_->control_rngs_, this->control_rngs_, C_DIM * sizeof(float2),
                                 cudaMemcpyHostToDevice, stream_));
    if (synchronize)
    {
      HANDLE_ERROR(cudaStreamSynchronize(stream_));
    }
  }
}

template <class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::GPUSetup()
{
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  if (!GPUMemStatus_)
  {
    model_d_ = Managed::GPUSetup(derived);
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;  // TODO should this be an exception?
  }
  derived->paramsToDevice();
}

template <class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::freeCudaMem()
{
  if (GPUMemStatus_)
  {
    cudaFree(model_d_);
    GPUMemStatus_ = false;
    model_d_ = nullptr;
  }
}
