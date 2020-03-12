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


template<class NETWORK_T, int S_DIM, int C_DIM>
__global__ void fullARNNTestKernel(NETWORK_T* model, float* state, float* control, float* state_der, float dt) {
  __shared__ float theta[NETWORK_T::SHARED_MEM_REQUEST_GRD + NETWORK_T::SHARED_MEM_REQUEST_BLK];
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  // calls enforce constraints -> compute state derivative -> increment state

  //printf("before enforceConstraints %d, %d\n", threadIdx.x, threadIdx.y);
  //printf("enforceConstraints %d, %d\n", threadIdx.x, threadIdx.y);
  //printf("before enforce Constraints %f, %f\n", (control+(tid*C_DIM))[0], (control+(tid*C_DIM))[1]);
  model->enforceConstraints(state+(tid*S_DIM), control+(tid*C_DIM));
  //printf("after enforce Constraints %f, %f\n", (control+(tid*C_DIM))[0], (control+(tid*C_DIM))[1]);
  model->computeStateDeriv(state+(tid*S_DIM), control+(tid*C_DIM), state_der+(tid*S_DIM), theta);
  // TODO generalize
  model->updateState(state+(tid*S_DIM), state_der+(tid*S_DIM), dt);
}

template<class NETWORK_T, int S_DIM, int C_DIM>
void launchFullARNNTestKernel(NETWORK_T& model, std::vector< std::array<float, S_DIM>>& state,
                              std::vector< std::array<float, C_DIM>>& control, std::vector< std::array<float, S_DIM>>& state_der,
                              float dt, int dim_y) {
  float* state_d;
  float* state_der_d;
  float* control_d;

  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float)*S_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&state_der_d, sizeof(float)*S_DIM))
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float)*C_DIM))

  HANDLE_ERROR(cudaMemcpy(state_d, state.data(), sizeof(float)*S_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(state_der_d, state_der.data(), sizeof(float)*S_DIM, cudaMemcpyHostToDevice))
  HANDLE_ERROR(cudaMemcpy(control_d, control.data(), sizeof(float)*C_DIM, cudaMemcpyHostToDevice))

  // make sure you cannot use invalid inputs
  dim3 threadsPerBlock(1, dim_y);
  dim3 numBlocks(1, 1);
  // launch kernel
  fullARNNTestKernel<NETWORK_T, S_DIM, C_DIM><<<numBlocks,threadsPerBlock>>>(model.model_d_, state_d, control_d, state_der_d, dt);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(state.data(), state_d, sizeof(float)*S_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(state_der.data(), state_der_d, sizeof(float)*S_DIM, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(control.data(), control_d, sizeof(float)*C_DIM, cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(state_d);
  cudaFree(state_der_d);
  cudaFree(control_d);
}


