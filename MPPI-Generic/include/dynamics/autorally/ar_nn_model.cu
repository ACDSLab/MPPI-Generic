template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(std::array<float2, C_DIM> control_rngs, cudaStream_t stream)
                  : Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams, S_DIM, C_DIM>(control_rngs, stream) {
  CPUSetup();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(cudaStream_t stream)
                  : Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams, S_DIM, C_DIM>(stream) {
  CPUSetup();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::~NeuralNetModel() {
  if(this->GPUMemStatus_) {
      freeCudaMem();
  }
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::freeCudaMem() {
  Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams, S_DIM, C_DIM>::freeCudaMem();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::CPUSetup(std::array<float2, C_DIM> control_rngs) {
  /*
  for(int i = 0; i < C_DIM; i++) {
    this->control_rngs_[i] = control_rngs[i];
  }
   */

  // setup the stride_idcs_ variable since it does not change in this template instantiation
  int stride = 0;
  // TODO verify that the change is correct
  for(int i = 0; i < NUM_LAYERS - 1; i++) {
    stride_idcs_[2 * i] = stride;
    stride += net_structure_[i+1] * net_structure_[i];
    stride_idcs_[2*i + 1] = stride;
    stride += net_structure_[i+1];
  }
  stride_idcs_[(NUM_LAYERS - 1)*2] = stride;
}

/*
template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::GPUSetup() {
  // allocate object
  if (!this->GPUMemStatus_) {
    // TODO check if this is setup properly in test cases
    this->model_d_ = Managed::GPUSetup(this);
  } else {
    std::cout << "GPU Memory already set." << std::endl;
  }
}
 */

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::updateModel(std::vector<int> description,
        std::vector<float> data) {
  for(int i = 0; i < description.size(); i++) {
    if(description[i] != net_structure_[i]) {
      std::cerr << "Invalid model trying to to be set for NN" << std::endl;
      exit(0);
    }
  }
  for (int i = 0; i < NUM_LAYERS - 1; i++){
    for (int j = 0; j < net_structure_[i+1]; j++){
      for (int k = 0; k < net_structure_[i]; k++){
        theta_[stride_idcs_[2*i] + j*net_structure_[i] + k] = data[stride_idcs_[2*i] + j*net_structure_[i] + k];
      }
    }
  }
  for (int i = 0; i < NUM_LAYERS - 1; i++){
    for (int j = 0; j < net_structure_[i+1]; j++){
      theta_[stride_idcs_[2*i + 1] + j] = data[stride_idcs_[2*i + 1] + j];
    }
  }
  if(this->GPUMemStatus_) {
    paramsToDevice();
  }
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::paramsToDevice() {
  // TODO copy to constant memory
  HANDLE_ERROR( cudaMemcpy(this->model_d_->control_rngs_, this->control_rngs_, NUM_PARAMS*sizeof(float), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->model_d_->theta_, theta_, NUM_PARAMS*sizeof(float), cudaMemcpyHostToDevice) );

#if defined(MPPI_NNET_USING_CONSTANT_MEM___) //Use constant memory.
  HANDLE_ERROR( cudaMemcpyToSymbol(NNET_PARAMS, theta_d_, NUM_PARAMS*sizeof(float)) );
#endif

}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::loadParams(const std::string& model_path) {
  int i,j,k;
  std::string bias_name = "";
  std::string weight_name = "";
  if (!fileExists(model_path)){
    std::cerr << "Could not load neural net model at path: " << model_path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(model_path);
  for (i = 0; i < NUM_LAYERS - 1; i++){
    // NN index from 1
    bias_name = "dynamics_b" + std::to_string(i + 1);
    weight_name = "dynamics_W" + std::to_string(i + 1);

    cnpy::NpyArray weight_i_raw = param_dict[weight_name];
    cnpy::NpyArray bias_i_raw = param_dict[bias_name];
    double* weight_i = weight_i_raw.data<double>();
    double* bias_i = bias_i_raw.data<double>();

    for (j = 0; j < net_structure_[i + 1]; j++){
      for (k = 0; k < net_structure_[i]; k++){
        // TODO why i - 1?
        theta_[stride_idcs_[2*i] + j*net_structure_[i] + k] = (float)weight_i[j*net_structure_[i] + k];
      }
    }
    for (j = 0; j < net_structure_[i+1]; j++){
      theta_[stride_idcs_[2*i + 1] + j] = (float)bias_i[j];
    }
  }
  //Save parameters to GPU memory
  paramsToDevice();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::enforceConstraints(
        float* state, float* control) {
  int i;
  for (i = 0; i < this->CONTROL_DIM; i++){
    if (control[i] < this->control_rngs_[i].x){
      control[i] = this->control_rngs_[i].x;
    } else if (control[i] > this->control_rngs_[i].y){
      control[i] = this->control_rngs_[i].y;
    }
  }
}

/*
template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeStateDeriv(
        float* state, float* control, float* state_der, float* theta_s) {
  // only propagate a single state, i.e. thread.y = 0
  // find the change in x,y,theta based off of the rest of the state
  if (threadIdx.y == 0){
    computeKinematics(state, state_der);
  }
  computeDynamics(state, control, state_der, theta_s);
}
 */

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeKinematics(
        float* state, float* state_der) {
  state_der[0] = cosf(state[2])*state[4] - sinf(state[2])*state[5];
  state_der[1] = sinf(state[2])*state[4] + cosf(state[2])*state[5];
  state_der[2] = -state[6]; //Pose estimate actually gives the negative yaw derivative
}

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
    curr_act[i] = state[i + (this->STATE_DIM - DYNAMICS_DIM)];
  }
  // iterate through the control to put into first layer
  for (i = tdy; i < this->CONTROL_DIM; i+= blockDim.y){
    curr_act[DYNAMICS_DIM + i] = control[i];
  }
  __syncthreads();
  // iterate through each layer
  for (i = 0; i < NUM_LAYERS - 1; i++){
    //Conditional compilation depending on if we're using a global constant memory array or not.
#if defined(MPPI_NNET_USING_CONSTANT_MEM___) //Use constant memory.
    W = &NNET_PARAMS[stride_idcs_[2*i]]; // weights
    b = &NNET_PARAMS[stride_idcs_[2*i + 1]]; // biases
#else //Use (slow) global memory.
    W = &theta_[stride_idcs_[2*i]]; // weights
    b = &theta_[stride_idcs_[2*i + 1]]; // biases
#endif
    // for first non input layer until last layer this thread deals with
    // calculates the next activation based on current
    for (j = tdy; j < net_structure_[i+1]; j += blockDim.y) {
      tmp = 0;
      // apply each neuron activation from current layer
      for (k = 0; k < net_structure_[i]; k++) {
        //No atomic add necessary.
        tmp += W[j*net_structure_[i] + k]*curr_act[k];
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
    state_der[i + (this->STATE_DIM - DYNAMICS_DIM)] = curr_act[i];
  }
  __syncthreads();
}

/*
template<int s_dim, int c_dim, int k_dim, int... layer_args>
__device__ void NeuralNetModel<s_dim, c_dim, k_dim, layer_args...>::updateState(
        float* state, float* state_der, float dt) {
  int i;
  int tdy = threadIdx.y;
  //Add the state derivative time dt to the current state.
  for (i = tdy; i < this->STATE_DIM; i+=blockDim.y){
    state[i] += state_der[i]*dt;
    state_der[i] = 0; //Important: reset the state derivative to zero.
  }
}
*/
