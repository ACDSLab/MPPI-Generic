template <typename CLASS_T, typename PARAMS_T>
__global__ void parameterTestKernel(CLASS_T* class_t, PARAMS_T& params)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0)
  {
    params = class_t->getParams();
  }
}

template <typename CLASS_T, typename PARAMS_T>
void launchParameterTestKernel(CLASS_T& class_t, PARAMS_T& params)
{
  PARAMS_T* params_d;
  HANDLE_ERROR(cudaMalloc((void**)&params_d, sizeof(PARAMS_T)))

  parameterTestKernel<CLASS_T, PARAMS_T><<<1, 1>>>(static_cast<CLASS_T*>(class_t.model_d_), *params_d);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(&params, params_d, sizeof(PARAMS_T), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(params_d);
}

template <typename DYNAMICS_T, int C_DIM>
__global__ void controlRangesTestKernel(DYNAMICS_T* dynamics, float2* control_rngs)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0)
  {
    float2* raw_ptr = dynamics->getControlRangesRaw();
    for (int i = 0; i < C_DIM; i++)
    {
      control_rngs[i].x = raw_ptr[i].x;
      control_rngs[i].y = raw_ptr[i].y;
    }
  }
}

template <typename DYNAMICS_T, int C_DIM>
void launchControlRangesTestKernel(DYNAMICS_T& dynamics, std::array<float2, C_DIM>& control_rngs)
{
  float2* ranges_d;
  HANDLE_ERROR(cudaMalloc((void**)&ranges_d, sizeof(float2) * control_rngs.size()))

  controlRangesTestKernel<DYNAMICS_T, C_DIM><<<1, 1>>>(static_cast<DYNAMICS_T*>(dynamics.model_d_), ranges_d);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(control_rngs.data(), ranges_d, sizeof(float2) * control_rngs.size(), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(ranges_d);
}

template <typename DYNAMICS_T, int S_DIM, int C_DIM>
__global__ void enforceConstraintTestKernel(DYNAMICS_T* dynamics, float* state, float* control, int num)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num)
  {
    dynamics->enforceConstraints(&state[tid * S_DIM], &control[tid * C_DIM]);
  }
}

template <typename DYNAMICS_T, int S_DIM, int C_DIM>
void launchEnforceConstraintTestKernel(DYNAMICS_T& dynamics, std::vector<std::array<float, S_DIM>>& state,
                                       std::vector<std::array<float, C_DIM>>& control, int dim_y)
{
  int count = state.size();
  float* state_d;
  float* control_d;
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * S_DIM * state.size()))
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float) * C_DIM * control.size()))

  HANDLE_ERROR(cudaMemcpy(state_d, state.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(control_d, control.data(), sizeof(float) * C_DIM * count, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(count, dim_y);
  dim3 numBlocks(1, 1);
  enforceConstraintTestKernel<DYNAMICS_T, S_DIM, C_DIM>
      <<<numBlocks, threadsPerBlock>>>(static_cast<DYNAMICS_T*>(dynamics.model_d_), state_d, control_d, count);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(state.data(), state_d, sizeof(float) * S_DIM * state.size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(control.data(), control_d, sizeof(float) * C_DIM * control.size(), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(control_d);
}

template <typename DYNAMICS_T, int S_DIM>
__global__ void updateStateTestKernel(DYNAMICS_T* dynamics, float* state, float* state_der, float dt, int num)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num)
  {
    dynamics->updateState(state + (tid * S_DIM), state_der + (tid * S_DIM), dt);
  }
}

template <typename DYNAMICS_T, int S_DIM>
void launchUpdateStateTestKernel(DYNAMICS_T& dynamics, std::vector<std::array<float, S_DIM>>& state,
                                 std::vector<std::array<float, S_DIM>>& state_der, float dt, int dim_y)
{
  int count = state.size();
  float* state_d;
  float* state_der_d;
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * S_DIM * state.size()))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float) * S_DIM * state_der.size()))

  HANDLE_ERROR(cudaMemcpy(state_d, state.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(count, dim_y);
  dim3 numBlocks(1, 1);
  updateStateTestKernel<DYNAMICS_T, S_DIM>
      <<<numBlocks, threadsPerBlock>>>(static_cast<DYNAMICS_T*>(dynamics.model_d_), state_d, state_der_d, dt, count);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(state.data(), state_d, sizeof(float) * S_DIM * state.size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(state_der.data(), state_der_d, sizeof(float) * S_DIM * state_der.size(), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
}

template <typename DYNAMICS_T, int S_DIM>
__global__ void computeKinematicsTestKernel(DYNAMICS_T* dynamics, float* state, float* state_der, int num)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num && threadIdx.y == 0)
  {
    dynamics->computeKinematics(state + (tid * S_DIM), state_der + (tid * S_DIM));
  }
}

template <typename DYNAMICS_T, int S_DIM>
void launchComputeKinematicsTestKernel(DYNAMICS_T& dynamics, std::vector<std::array<float, S_DIM>>& state,
                                       std::vector<std::array<float, S_DIM>>& state_der, int dim_y)
{
  int count = state.size();
  float* state_d;
  float* state_der_d;
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * S_DIM * state.size()))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float) * S_DIM * state_der.size()))

  HANDLE_ERROR(cudaMemcpy(state_d, state.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(count, dim_y);
  dim3 numBlocks(1, 1);
  computeKinematicsTestKernel<DYNAMICS_T, S_DIM>
      <<<numBlocks, threadsPerBlock>>>(static_cast<DYNAMICS_T*>(dynamics.model_d_), state_d, state_der_d, count);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(state.data(), state_d, sizeof(float) * S_DIM * state.size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(state_der.data(), state_der_d, sizeof(float) * S_DIM * state_der.size(), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
}

template <class DYNAMICS_T, int S_DIM, int C_DIM>
__global__ void computeDynamicsTestKernel(DYNAMICS_T* model, float* state, float* control, float* state_der, int count)
{
  __shared__ float theta[DYNAMICS_T::SHARED_MEM_REQUEST_GRD + DYNAMICS_T::SHARED_MEM_REQUEST_BLK];

  model->computeDynamics(state, control, state_der, theta);
}

template <class DYNAMICS_T, int S_DIM, int C_DIM>
void launchComputeDynamicsTestKernel(DYNAMICS_T& dynamics, std::vector<std::array<float, S_DIM>>& state,
                                     std::vector<std::array<float, C_DIM>>& control,
                                     std::vector<std::array<float, S_DIM>>& state_der, int dim_y)
{
  int count = state.size();
  float* state_d;
  float* state_der_d;
  float* control_d;

  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * S_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float) * S_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float) * C_DIM))

  HANDLE_ERROR(cudaMemcpy(state_d, state.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(control_d, control.data(), sizeof(float) * C_DIM * count, cudaMemcpyHostToDevice));

  // make sure you cannot use invalid inputs
  dim3 threadsPerBlock(1, dim_y);
  dim3 numBlocks(1, 1);
  // launch kernel
  computeDynamicsTestKernel<DYNAMICS_T, S_DIM, C_DIM>
      <<<numBlocks, threadsPerBlock>>>(dynamics.model_d_, state_d, control_d, state_der_d, count);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(state.data(), state_d, sizeof(float) * S_DIM * state.size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(state_der.data(), state_der_d, sizeof(float) * S_DIM * state_der.size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(control.data(), control_d, sizeof(float) * C_DIM * control.size(), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
  cudaFree(control_d);
}

template <typename DYNAMICS_T, int S_DIM, int C_DIM>
__global__ void computeStateDerivTestKernel(DYNAMICS_T* dynamics, float* state, float* control, float* state_der,
                                            int num)
{
  __shared__ float theta[DYNAMICS_T::SHARED_MEM_REQUEST_GRD + DYNAMICS_T::SHARED_MEM_REQUEST_BLK];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num)
  {
    // printf("calling on thread %d, %d\n", tid, threadIdx.y);
    dynamics->computeStateDeriv(state + (tid * S_DIM), control + (tid * C_DIM), state_der + (tid * S_DIM), theta);
  }
}

template <typename DYNAMICS_T, int S_DIM, int C_DIM>
void launchComputeStateDerivTestKernel(DYNAMICS_T& dynamics, std::vector<std::array<float, S_DIM>>& state,
                                       std::vector<std::array<float, C_DIM>>& control,
                                       std::vector<std::array<float, S_DIM>>& state_der, int dim_y)
{
  int count = state.size();
  float* state_d;
  float* control_d;
  float* state_der_d;
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * S_DIM * state.size()))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float) * S_DIM * state_der.size()))
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float) * C_DIM * control.size()))

  HANDLE_ERROR(cudaMemcpy(state_d, state.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der.data(), sizeof(float) * S_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(control_d, control.data(), sizeof(float) * C_DIM * count, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(count, dim_y);
  dim3 numBlocks(1, 1);
  computeStateDerivTestKernel<DYNAMICS_T, S_DIM, C_DIM><<<numBlocks, threadsPerBlock>>>(
      static_cast<DYNAMICS_T*>(dynamics.model_d_), state_d, control_d, state_der_d, count);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(cudaMemcpy(state.data(), state_d, sizeof(float) * S_DIM * state.size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(state_der.data(), state_der_d, sizeof(float) * S_DIM * state_der.size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(control.data(), control_d, sizeof(float) * C_DIM * control.size(), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
  cudaFree(control_d);
}
