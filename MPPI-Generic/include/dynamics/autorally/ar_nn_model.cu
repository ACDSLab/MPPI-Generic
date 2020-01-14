template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(float delta_t, std::array<float2, C_DIM> control_rngs, cudaStream_t stream) {
  CPUSetup(delta_t, control_rngs, stream);
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(float delta_t, cudaStream_t stream) {
  std::array<float2, C_DIM> control_rngs;
  for(int i = 0; i < C_DIM; i++) {
    control_rngs[i].x = -FLT_MAX;
    control_rngs[i].y = FLT_MAX;
  }
  CPUSetup(delta_t, control_rngs, stream);
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::~NeuralNetModel() {

}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::CPUSetup(float delta_t, std::array<float2, C_DIM> control_rngs, cudaStream_t stream) {
  this->bindToStream(stream);
  this->dt_ = delta_t;
  for(int i = 0; i < C_DIM; i++) {
    control_rngs_[i] = control_rngs[i];
  }

  // setup the stride_idcs_ variable since it does not change in this template instantiation
  int stride = 0;
  // TODO what?
  for(int i = 0; i < NUM_LAYERS - 1; i++) {
    stride_idcs_[2 * i] = stride;
    stride += net_structure_[i+1] * net_structure_[i];
    stride_idcs_[2*i + 1] = stride;
    stride += net_structure_[i+1];
  }
  stride_idcs_[(NUM_LAYERS - 1)*2] = stride;
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::GPUSetup() {
  // allocate object
  if (!this->GPUMemStatus_) {
    // TODO check if this is setup properly in test cases
    model_d_ = Managed::GPUSetup(this);
  } else {
    std::cout << "GPU Memory already set." << std::endl;
  }
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::paramsToDevice() {

}


/*
template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeDynamics(float* state, float* control, float* state_der, float* theta_s)
{
  float* curr_act;
  float* next_act;
  float* tmp_act;
  float tmp;
  float* W;
  float* b;
  int tdx = threadIdx.x;
  int tdy = threadIdx.y;
  int tdz = threadIdx.z;
  int i,j,k;
  curr_act = &theta_s[(2*LARGEST_LAYER)*(blockDim.x*tdz + tdx)];
  next_act = &theta_s[(2*LARGEST_LAYER)*(blockDim.x*tdz + tdx) + LARGEST_LAYER];
  // iterate through the part of the state that should be an input to the NN
  for (i = tdy; i < DYNAMICS_DIM; i+= blockDim.y){
    curr_act[i] = state[i + (STATE_DIM - DYNAMICS_DIM)];
  }
  // iterate through the control to put into first layer
  for (i = tdy; i < CONTROL_DIM; i+= blockDim.y){
    curr_act[DYNAMICS_DIM + i] = control[i];
  }
  __syncthreads();
  // iterate through each layer
  for (i = 0; i < NUM_LAYERS - 1; i++){
    //Conditional compilation depending on if we're using a global constant memory array or not.
#if defined(MPPI_NNET_USING_CONSTANT_MEM___) //Use constant memory.
    W = &NNET_PARAMS[stride_idcs_d_[2*i]]; // weights
    b = &NNET_PARAMS[stride_idcs_d_[2*i + 1]]; // biases
#else //Use (slow) global memory.
    W = &theta_d_[stride_idcs_d_[2*i]]; // weights
    b = &theta_d_[stride_idcs_d_[2*i + 1]]; // biases
#endif
    // for first non input layer until last layer this thread deals with
    // calculates the next activation based on current
    for (j = tdy; j < net_structure_d_[i+1]; j += blockDim.y) {
      tmp = 0;
      // apply each neuron activation from current layer
      for (k = 0; k < net_structure_d_[i]; k++) {
        //No atomic add necessary.
        tmp += W[j*net_structure_d_[i] + k]*curr_act[k];
      }
      // add bias from next layer and neuron
      tmp += b[j];
      if (i < NUM_LAYERS - 2){
        tmp = MPPI_NNET_NONLINEARITY(tmp);
      }
      next_act[j] = tmp;
    }
    //Swap the two pointers
    tmp_act = curr_act;
    curr_act = next_act;
    next_act = tmp_act;
    __syncthreads();
  }
  // copies results back into state derivative
  for (i = tdy; i < DYNAMICS_DIM; i+= blockDim.y){
    state_der[i + (STATE_DIM - DYNAMICS_DIM)] = curr_act[i];
  }
  __syncthreads();
}
 */

