template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(std::array<float2, C_DIM> control_rngs, cudaStream_t stream)
                  : Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>(control_rngs, stream) {
  CPUSetup();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(cudaStream_t stream)
                  : Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>(stream) {
  CPUSetup();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::~NeuralNetModel() {
  if(weights_ != nullptr) {
    delete[] weights_;
  }
  if(biases_ != nullptr) {
    delete[] biases_;
  }
  if(weighted_in_ != nullptr) {
    delete[] weighted_in_;
  }
  if(this->GPUMemStatus_) {
    freeCudaMem();
  }
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::freeCudaMem() {
  Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>::freeCudaMem();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::CPUSetup() {
  // setup the CPU side values
  weights_ = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>[NUM_LAYERS-1];
  biases_ = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>[NUM_LAYERS-1];

  weighted_in_ = new Eigen::MatrixXf[NUM_LAYERS - 1];
  for(int i = 1; i < NUM_LAYERS; i++) {
    weighted_in_[i-1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], 1);
    weights_[i-1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], this->params_.net_structure[i-1]);
    biases_[i-1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], 1);
  }
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::updateModel(std::vector<int> description,
        std::vector<float> data) {
  for(int i = 0; i < description.size(); i++) {
    if(description[i] != this->params_.net_structure[i]) {
      std::cerr << "Invalid model trying to to be set for NN" << std::endl;
      exit(0);
    }
  }
  for (int i = 0; i < NUM_LAYERS - 1; i++){
    for (int j = 0; j < this->params_.net_structure[i+1]; j++){
      for (int k = 0; k < this->params_.net_structure[i]; k++){
        weights_[i](j,k) = data[this->params_.stride_idcs[2*i] + j*this->params_.net_structure[i] + k];
        this->params_.theta[this->params_.stride_idcs[2*i] + j*this->params_.net_structure[i] + k] = data[this->params_.stride_idcs[2*i] + j*this->params_.net_structure[i] + k];
      }
    }
  }
  for (int i = 0; i < NUM_LAYERS - 1; i++){
    for (int j = 0; j < this->params_.net_structure[i+1]; j++){
      biases_[i](j,0) = data[this->params_.stride_idcs[2*i + 1] + j];
      this->params_.theta[this->params_.stride_idcs[2*i + 1] + j] = data[this->params_.stride_idcs[2*i + 1] + j];
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
  HANDLE_ERROR( cudaMemcpy(this->model_d_->params_.theta, this->params_.theta, NUM_PARAMS*sizeof(float), cudaMemcpyHostToDevice) );

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

    for (j = 0; j < this->params_.net_structure[i + 1]; j++){
      for (k = 0; k < this->params_.net_structure[i]; k++){
        // TODO why i - 1?
        this->params_.theta[this->params_.stride_idcs[2*i] + j*this->params_.net_structure[i] + k] = (float)weight_i[j*this->params_.net_structure[i] + k];
        weights_[i](j,k) = (float)weight_i[j*this->params_.net_structure[i] + k];
      }
    }
    for (j = 0; j < this->params_.net_structure[i+1]; j++){
      this->params_.theta[this->params_.stride_idcs[2*i + 1] + j] = (float)bias_i[j];
      biases_[i](j,0) = (float)bias_i[j];
    }
  }
  //Save parameters to GPU memory
  paramsToDevice();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeGrad(const Eigen::Ref<const state_array>& state,
                                                                     const Eigen::Ref<const control_array>& control,
                                                                     Eigen::Ref<dfdx> A,
                                                                     Eigen::Ref<dfdu> B) {
  // TODO results are not returned
  Eigen::Matrix<float, S_DIM, S_DIM + C_DIM> jac;
  jac.setZero();

  //Start with the kinematic and physics model derivatives
  jac.row(0) << 0, 0, -sin(state(2))*state(4) - cos(state(2))*state(5), 0, cos(state(2)), -sin(state(2)), 0, 0, 0;
  jac.row(1) << 0, 0, cos(state(2))*state(4) - sin(state(2))*state(5), 0, sin(state(2)), cos(state(2)), 0, 0, 0;
  jac.row(2) << 0, 0, 0, 0, 0, 0, -1, 0, 0;

  Eigen::MatrixXf state_der(S_DIM, 1);

  //First do the forward pass
  computeDynamics(state, control, state_der);

  //Start backprop
  Eigen::MatrixXf ip_delta = Eigen::MatrixXf::Identity(DYNAMICS_DIM, DYNAMICS_DIM);
  Eigen::MatrixXf temp_delta = Eigen::MatrixXf::Identity(DYNAMICS_DIM, DYNAMICS_DIM);

  //Main backprop loop
  for (int i = NUM_LAYERS-2; i > 0; i--){
    Eigen::MatrixXf zp = weighted_in_[i-1];
    for (int j = 0; j < this->params_.net_structure[i]; j++){
      zp(j) = MPPI_NNET_NONLINEARITY_DERIV(zp(j));
    }
    ip_delta =  ( (weights_[i]).transpose()*ip_delta).eval();
    for (int j = 0; j < DYNAMICS_DIM; j++){
      ip_delta.col(j) = ip_delta.col(j).array() * zp.array();
    }
  }
  //Finish the backprop loop
  ip_delta = ( ((weights_[0]).transpose())*ip_delta).eval();
  jac.bottomRightCorner(DYNAMICS_DIM, DYNAMICS_DIM + C_DIM) += ip_delta.transpose();
  A = jac.leftCols(S_DIM);
  B = jac.rightCols(C_DIM);
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeKinematics(const Eigen::Ref<const state_array>& state,
        Eigen::Ref<state_array> state_der) {
  state_der(0) = cosf(state(2))*state(4) - sinf(state(2))*state(5);
  state_der(1) = sinf(state(2))*state(4) + cosf(state(2))*state(5);
  state_der(2) = -state(6); //Pose estimate actually gives the negative yaw derivative
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeDynamics(const Eigen::Ref<const state_array>& state,
        const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der) {
  int i,j;
  Eigen::MatrixXf acts(this->params_.net_structure[0], 1);
  for (i = 0; i < DYNAMICS_DIM; i++){
    acts(i) = state(i + (S_DIM - DYNAMICS_DIM));
  }
  for (i = 0; i < C_DIM; i++){
    acts(DYNAMICS_DIM + i) = control(i);
  }
  for (i = 0; i < NUM_LAYERS - 1; i++){
    weighted_in_[i] = (weights_[i]*acts + biases_[i]).eval();
    acts = Eigen::MatrixXf::Zero(this->params_.net_structure[i+1], 1);
    if (i < NUM_LAYERS - 2) { //Last layer doesn't apply any non-linearity
      for (j = 0; j < this->params_.net_structure[i+1]; j++){
        acts(j) = MPPI_NNET_NONLINEARITY( (weighted_in_[i])(j) ); //Nonlinear component.
      }
    }
    else {
      for (j = 0; j < this->params_.net_structure[i+1]; j++){
        acts(j) = (weighted_in_[i])(j) ;
      }
    }
  }
  for (i = 0; i < DYNAMICS_DIM; i++){
    state_der(i + (S_DIM - DYNAMICS_DIM)) = acts(i);
  }
}

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
    curr_act[i] = state[i + (S_DIM - DYNAMICS_DIM)];
  }
  // iterate through the control to put into first layer
  for (i = tdy; i < C_DIM; i+= blockDim.y){
    curr_act[DYNAMICS_DIM + i] = control[i];
  }
  __syncthreads();
  // iterate through each layer
  for (i = 0; i < NUM_LAYERS - 1; i++){
    //Conditional compilation depending on if we're using a global constant memory array or not.
#if defined(MPPI_NNET_USING_CONSTANT_MEM___) //Use constant memory.
    W = &NNET_PARAMS[this->params_.stride_idcs[2*i]]; // weights
    b = &NNET_PARAMS[this->params_.stride_idcs[2*i + 1]]; // biases
#else //Use (slow) global memory.
    W = &this->params_.theta[this->params_.stride_idcs[2*i]]; // weights
    b = &this->params_.theta[this->params_.stride_idcs[2*i + 1]]; // biases
#endif
    // for first non input layer until last layer this thread deals with
    // calculates the next activation based on current
    for (j = tdy; j < this->params_.net_structure[i+1]; j += blockDim.y) {
      tmp = 0;
      // apply each neuron activation from current layer
      for (k = 0; k < this->params_.net_structure[i]; k++) {
        //No atomic add necessary.
        tmp += W[j*this->params_.net_structure[i] + k]*curr_act[k];
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
    state_der[i + (S_DIM - DYNAMICS_DIM)] = curr_act[i];
  }
  __syncthreads();
}

//template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
//__device__ void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeStateDeriv(float* state, float* control, float* state_der, float* theta_s) {
//  // only propagate a single state, i.e. thread.y = 0
//  // find the change in x,y,theta based off of the rest of the state
//  if (threadIdx.y == 0){
//    //printf("state at 0 before kin: %f\n", state[0]);
//    computeKinematics(state, state_der);
//    //printf("state at 0 after kin: %f\n", state[0]);
//  }
//  computeDynamics(state, control, state_der, theta_s);
//  //printf("state at 0 after dyn: %f\n", state[0]);
//}

