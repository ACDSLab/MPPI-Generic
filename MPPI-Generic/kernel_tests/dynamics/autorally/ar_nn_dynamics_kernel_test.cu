template <class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
__global__ void parameterCheckTestKernel(NETWORK_T* model,  float* theta, int* stride, int* net_structure) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid == 0) {
    for(int i = 0; i < THETA_SIZE; i++) {
      theta[i] = model->getThetaPtr()[i];
    }
    for(int i = 0; i < STRIDE_SIZE; i++) {
      stride[i] = model->getStrideIdcsPtr()[i];
    }
    for(int i = 0; i < NUM_LAYERS; i++) {
      net_structure[i] = model->getNetStructurePtr()[i];
    }
  }
}

template<class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
void launchParameterCheckTestKernel(NETWORK_T& model, std::array<float, THETA_SIZE>& theta, std::array<int, STRIDE_SIZE>& stride,
                                    std::array<int, NUM_LAYERS>& net_structure) {
  float* theta_d;
  int* stride_d;
  int* net_structure_d;

  HANDLE_ERROR(cudaMalloc((void**)&theta_d, sizeof(float)*theta.size()))
  HANDLE_ERROR(cudaMalloc((void**)&stride_d, sizeof(int)*stride.size()))
  HANDLE_ERROR(cudaMalloc((void**)&net_structure_d, sizeof(int)*net_structure.size()))

  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks(1, 1);
  parameterCheckTestKernel<NETWORK_T, THETA_SIZE, STRIDE_SIZE, NUM_LAYERS><<<numBlocks,threadsPerBlock>>>(model.model_d_, theta_d, stride_d, net_structure_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(theta.data(), theta_d, sizeof(float)*theta.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(stride.data(), stride_d, sizeof(int)*stride.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(net_structure.data(), net_structure_d, sizeof(int)*net_structure.size(), cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(theta_d);
  cudaFree(stride_d);
  cudaFree(net_structure_d);
}

template<class NETWORK_T, int STATE_DIM>
__global__ void incrementStateTestKernel(NETWORK_T* model, float* state, float* state_der) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid == 0) {
    // TODO generalize
    model->updateState(state, state_der, 0.1);
  }
}

template<class NETWORK_T, int BLOCK_DIM_Y, int STATE_DIM>
void launchIncrementStateTestKernel(NETWORK_T& model, std::array<float, STATE_DIM>& state, std::array<float, 7>& state_der) {
  float* state_d;
  float* state_der_d;

  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float)*state.size()))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float)*state_der.size()))

  HANDLE_ERROR(cudaMemcpy(state_d, state.data() , sizeof(float)*state.size(), cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der.data(), sizeof(float)*state_der.size(), cudaMemcpyHostToDevice))

  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks(1, 1);
  // launch kernel
  incrementStateTestKernel<NETWORK_T, STATE_DIM><<<numBlocks,threadsPerBlock>>>(model.model_d_, state_d, state_der_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(state.data(), state_d, sizeof(float)*state.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(state_der.data(), state_der_d, sizeof(float)*state_der.size(), cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
}

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM>
__global__ void computeDynamicsTestKernel(NETWORK_T* model, float* state, float* control, float* state_der) {
  __shared__ float theta[NETWORK_T::SHARED_MEM_REQUEST_GRD + NETWORK_T::SHARED_MEM_REQUEST_BLK];

  model->computeDynamics(state, control, state_der, theta);
}

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM, int BLOCKSIZE_Y>
void launchComputeDynamicsTestKernel(NETWORK_T& model, float* state, float* control, float* state_der) {
  float* state_d;
  float* state_der_d;
  float* control_d;

  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float)*STATE_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float)*STATE_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float)*CONTROL_DIM))

  HANDLE_ERROR(cudaMemcpy(state_d, state , sizeof(float)*STATE_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der, sizeof(float)*STATE_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(control_d, control, sizeof(float)*CONTROL_DIM, cudaMemcpyHostToDevice))

  // make sure you cannot use invalid inputs
  dim3 threadsPerBlock(1, BLOCKSIZE_Y);
  dim3 numBlocks(1, 1);
  // launch kernel
  computeDynamicsTestKernel<NETWORK_T, STATE_DIM, CONTROL_DIM><<<numBlocks,threadsPerBlock>>>(model.model_d_, state_d, control_d, state_der_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(state, state_d, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(state_der, state_der_d, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(control, control_d, sizeof(float)*CONTROL_DIM, cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
  cudaFree(control_d);
}

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM>
__global__ void computeStateDerivTestKernel(NETWORK_T* model, float* state, float* control, float* state_der) {
  __shared__ float theta[NETWORK_T::SHARED_MEM_REQUEST_GRD + NETWORK_T::SHARED_MEM_REQUEST_BLK];

  model->computeStateDeriv(state, control, state_der, theta);
}

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM, int BLOCKSIZE_Y>
void launchComputeStateDerivTestKernel(NETWORK_T& model, float* state, float* control, float* state_der) {
  float* state_d;
  float* state_der_d;
  float* control_d;

  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float)*STATE_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float)*STATE_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float)*CONTROL_DIM))

  HANDLE_ERROR(cudaMemcpy(state_d, state , sizeof(float)*STATE_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der, sizeof(float)*STATE_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(control_d, control, sizeof(float)*CONTROL_DIM, cudaMemcpyHostToDevice))

  // make sure you cannot use invalid inputs
  dim3 threadsPerBlock(1, BLOCKSIZE_Y);
  dim3 numBlocks(1, 1);
  // launch kernel
  computeStateDerivTestKernel<NETWORK_T, STATE_DIM, CONTROL_DIM><<<numBlocks, threadsPerBlock>>>(model.model_d_, state_d, control_d, state_der_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(state, state_d, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(state_der, state_der_d, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(control, control_d, sizeof(float)*CONTROL_DIM, cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
  cudaFree(control_d);
}

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM>
__global__ void fullARNNTestKernel(NETWORK_T* model, float* state, float* control, float* state_der) {
  __shared__ float theta[NETWORK_T::SHARED_MEM_REQUEST_GRD + NETWORK_T::SHARED_MEM_REQUEST_BLK];

  int tdy = threadIdx.y;
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  // calls enforce constraints -> compute state derivative -> increment state
  if(tdy == 0) {
    model->enforceConstraints(state, control);
  }
  model->computeStateDeriv(state, control, state_der, theta);
  // TODO generalize
  model->updateState(state, state_der, 0.1);
}

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM, int BLOCKSIZE_Y>
void launchFullARNNTestKernel(NETWORK_T& model, float* state, float* control, float* state_der) {
  float* state_d;
  float* state_der_d;
  float* control_d;

  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float)*STATE_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float)*STATE_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float)*CONTROL_DIM))

  HANDLE_ERROR(cudaMemcpy(state_d, state , sizeof(float)*STATE_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der, sizeof(float)*STATE_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(control_d, control, sizeof(float)*CONTROL_DIM, cudaMemcpyHostToDevice))

  // make sure you cannot use invalid inputs
  dim3 threadsPerBlock(1, BLOCKSIZE_Y);
  dim3 numBlocks(1, 1);
  // launch kernel
  fullARNNTestKernel<NETWORK_T, STATE_DIM, CONTROL_DIM><<<numBlocks, threadsPerBlock>>>(model.model_d_, state_d, control_d, state_der_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(state, state_d, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(state_der, state_der_d, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(control, control_d, sizeof(float)*CONTROL_DIM, cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
  cudaFree(control_d);
}


