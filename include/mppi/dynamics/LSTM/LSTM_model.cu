template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::LSTMModel(std::array<float2, C_DIM> control_rngs, cudaStream_t stream)
                  : Dynamics<LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>, LSTMDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>(control_rngs, stream) {
  CPUSetup();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::LSTMModel(cudaStream_t stream)
                  : Dynamics<LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>, LSTMDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>(stream) {
  CPUSetup();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::~LSTMModel() {
  if(weights_ != nullptr) {
    delete[] weights_;
  }
  if(biases_ != nullptr) {
    delete[] biases_;
  }
  if(weighted_in_ != nullptr) {
    delete[] weighted_in_;
  }
  freeCudaMem();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::freeCudaMem() {
  Dynamics<LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>, LSTMDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>::freeCudaMem();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::CPUSetup() {
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
void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::updateModel(std::vector<int> description,
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
void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::paramsToDevice() {
  // TODO copy to constant memory
  HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->control_rngs_,
                           this->control_rngs_,
                           NUM_PARAMS*sizeof(float), cudaMemcpyHostToDevice,
                           this->stream_) );

  if (this->params_.copy_everything) {
    // Copy Weight Matrices
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_im,
                                  this->params_.W_im,
                                  this->params_.HIDDEN_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_fm,
                                  this->params_.W_fm,
                                  this->params_.HIDDEN_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_om,
                                  this->params_.W_om,
                                  this->params_.HIDDEN_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_cm,
                                  this->params_.W_cm,
                                  this->params_.HIDDEN_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_ii,
                                  this->params_.W_ii,
                                  this->params_.STATE_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_fi,
                                  this->params_.W_fi,
                                  this->params_.STATE_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_oi,
                                  this->params_.W_oi,
                                  this->params_.STATE_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_ci,
                                  this->params_.W_ci,
                                  this->params_.STATE_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.W_y,
                                  this->params_.W_y,
                                  this->params_.STATE_HIDDEN_SIZE*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    // Copy bias matrices
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.b_i,
                                  this->params_.b_i,
                                  this->params_.HIDDEN_DIM*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.b_f,
                                  this->params_.b_f,
                                  this->params_.HIDDEN_DIM*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.b_o,
                                  this->params_.b_o,
                                  this->params_.HIDDEN_DIM*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.b_c,
                                  this->params_.b_c,
                                  this->params_.HIDDEN_DIM*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.b_y,
                                  this->params_.b_y,
                                  this->params_.DYNAMICS_DIM*sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_) );
    this->params_.copy_everything = false;
  }
  if (this->params_.update_buffer) {
    this->params_.updateBuffer();
  }
  this->params_.updateInitialLSTMState();
  HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.initial_hidden,
                                this->params_.initial_hidden,
                                this->params_.HIDDEN_DIM*sizeof(float),
                                cudaMemcpyHostToDevice, this->stream_) );
  HANDLE_ERROR( cudaMemcpyAsync(this->model_d_->params_.initial_cell,
                                this->params_.initial_cell,
                                this->params_.HIDDEN_DIM*sizeof(float),
                                cudaMemcpyHostToDevice, this->stream_) );
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::loadParams(const std::string& model_path) {
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

    // copy over the weights
    for (j = 0; j < this->params_.net_structure[i + 1]; j++){
      for (k = 0; k < this->params_.net_structure[i]; k++){
        // TODO why i - 1?
        this->params_.theta[this->params_.stride_idcs[2*i] + j*this->params_.net_structure[i] + k] =
                (float)weight_i[j*this->params_.net_structure[i] + k];
        weights_[i](j,k) = (float)weight_i[j*this->params_.net_structure[i] + k];
      }
    }
    // copy over the bias
    for (j = 0; j < this->params_.net_structure[i+1]; j++){
      this->params_.theta[this->params_.stride_idcs[2*i + 1] + j] = (float)bias_i[j];
      biases_[i](j,0) = (float)bias_i[j];
    }
  }
  //Save parameters to GPU memory
  paramsToDevice();
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
bool LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeGrad(const Eigen::Ref<const state_array>& state,
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

  state_array state_der;

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
  return true;
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeKinematics(const Eigen::Ref<const state_array>& state,
        Eigen::Ref<state_array> state_der) {
  state_der(0) = cosf(state(2))*state(4) - sinf(state(2))*state(5);
  state_der(1) = sinf(state(2))*state(4) + cosf(state(2))*state(5);
  state_der(2) = -state(6); //Pose estimate actually gives the negative yaw derivative
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeDynamics(const Eigen::Ref<const state_array>& state,
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
__device__ void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeKinematics(
        float* state, float* state_der) {
  state_der[0] = cosf(state[2])*state[4] - sinf(state[2])*state[5];
  state_der[1] = sinf(state[2])*state[4] + cosf(state[2])*state[5];
  state_der[2] = -state[6]; //Pose estimate actually gives the negative yaw derivative
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeDynamics(float* state, float* control, float* state_der, float* theta_s)
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
#if defined(MPPI_NNET_USING_CONSTANT_MEM__) //Use constant memory.
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

// x = v_k
// h = m_{k-1}
// c = c_{k-1}
template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void LSTMModel<S_DIM, C_DIM, K_DIM, layer_args...>::LSTMLayer(float* x, float* theta_s,
                                                                         float* c,
                                                                         float* h,
                                                                         float* next_output,
                                                                         float* next_cell_state,
                                                                         float* next_hidden_state) {
  // float* curr_act;
  // float* next_act;
  // float* tmp_act;
  float tmp;
  // Weights
  float* W_ii = &(this->params_.W_ii);
  float* W_im = &(this->params_.W_im);
  float* W_fi = &(this->params_.W_fi);
  float* W_fm = &(this->params_.W_fm);
  float* W_oi = &(this->params_.W_oi);
  float* W_om = &(this->params_.W_om);
  float* W_ci = &(this->params_.W_ci);
  float* W_cm = &(this->params_.W_cm);

  // float* W_im ; // hidden * state, hidden * hidden
  // float* W_fi, *W_fm; // hidden * state, hidden * hidden
  // float* W_oi, *W_om; // hidden * state, hidden * hidden
  // float* W_ci, *W_cm; // hidden * state, hidden * hidden
  // Biases
  float* b_i = &(this->params_.b_i); // hidden_size
  float* b_f = &(this->params_.b_f); // hidden_size
  float* b_o = &(this->params_.b_o); // hidden_size
  float* b_c = &(this->params_.b_c); // hidden_size
  // Intermediate outputs
  float* c = &theta_s[0];
  float* h = &theta_s[this->params_.HIDDEN_SIZE];
  float* next_cell_state = &theta_s[2 * this->params_.HIDDEN_SIZE];
  float* next_hidden_state = &theta_s[3 * this->params_.HIDDEN_SIZE];
  float* g_ki = &theta_s[4 * this->params_.HIDDEN_SIZE];
  float* g_kf = &theta_s[5 * this->params_.HIDDEN_SIZE];
  float* g_ko = &theta_s[6 * this->params_.HIDDEN_SIZE];
  float* cell_update = &theta_s[7 * this->params_.HIDDEN_SIZE];
  float* intermediate_y = &theta_s[8 * this->params_.HIDDEN_SIZE];
  // float* g_ki, *g_kf, *g_ko, *cell_update; // hidden_size

  float dt = 0.01;
  float* W_y,; // state * hidden
  float* b_y,; // state
  int i,j,k;
  int input_size, hidden_size;

  // float* intermediate_y;
  int index = 0;
  // input gate
  for (i = 0; i < hidden_size; i++) {
    g_ki[i] = 0;
    for (j = 0; j < input_size; j++) {
      g_ki[i] += W_ii[i * input_size + j] * x[j];
    }
    for (j = 0; j < hidden_size; j++) {
      index = i * hidden_size + j;
      g_ki[i] += W_im[index] * h[j];
    }
    g_ki[i] += b_i[i];
    g_ki[i] = SIGMOID(g_ki[i]);
  }
  // forget gate
  for (i = 0; i < hidden_size; i++) {
    g_kf[i] = 0;
    for (j = 0; j < input_size; j++) {
      g_kf[i] += W_fi[i * input_size + j] * x[j];
    }
    for (j = 0; j < hidden_size; j++) {
      index = i * hidden_size + j;
      g_kf[i] += W_fm[index] * h[j];
    }
    g_kf[i] += b_f[i];
    g_kf[i] = SIGMOID(g_kf[i]);
  }
  // output gate
  for (i = 0; i < hidden_size; i++) {
    g_ko[i] = 0;
    for (j = 0; j < input_size; j++) {
      g_ko[i] += W_oi[i * input_size + j] * x[j];
    }
    for (j = 0; j < hidden_size; j++) {
      index = i * hidden_size + j;
      g_ko[i] += W_om[index] * h[j];
    }
    g_ko[i] += b_o[i];
    g_ko[i] = SIGMOID(g_ko[i]);
  }
  // cell update
  for (i = 0; i < hidden_size; i++) {
    cell_update[i] = 0;
    for (j = 0; j < input_size; j++) {
      cell_update[i] += W_ci[i * input_size + j] * x[j];
    }
    for (j = 0; j < hidden_size; j++) {
      cell_update[i] += W_cm[i * hidden_size + j] * h[j];
    }
    cell_update[i] += b_c[i];
    cell_update[i] = RELU(cell_update[i]);
  }

  // outputs
  for (i = 0; i < hidden_size; i++) {
    next_cell_state[i] = g_ki[i] * cell_update[i] + g_kf[i] * c[j];
  }
  for (i = 0; i < hidden_size; i++) {
    next_hidden_state[i] = tanhf(next_cell_state[i]) * g_ko[i];
  }

  for (i = 0; i < input_size; i++) {
    intermediate_y = 0;
    for (j = 0; j < hidden_size; j++) {
      intermediate_y += W_y1[i * hidden_size + j] * next_hidden_state[j];
    }
    intermediate_y += b_y1[i];
    intermediate_y[i] = intermediate_y;
  }
}