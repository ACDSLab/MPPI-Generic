template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::LSTMModel(std::array<float2, C_DIM> control_rngs,
                                                                   cudaStream_t stream)
  : PARENT_CLASS(control_rngs, stream)
{
  CPUSetup();
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::LSTMModel(cudaStream_t stream) : PARENT_CLASS(stream)
{
  CPUSetup();
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::~LSTMModel()
{
  // if(weights_ != nullptr) {
  //   delete[] weights_;
  // }
  // if(biases_ != nullptr) {
  //   delete[] biases_;
  // }
  // if(weighted_in_ != nullptr) {
  //   delete[] weighted_in_;
  // }
  freeCudaMem();
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::freeCudaMem()
{
  PARENT_CLASS::freeCudaMem();
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::CPUSetup()
{
  // setup the CPU side values
  // weights_ = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>[NUM_LAYERS-1];
  // biases_ = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>[NUM_LAYERS-1];

  // weighted_in_ = new Eigen::MatrixXf[NUM_LAYERS - 1];
  // for(int i = 1; i < NUM_LAYERS; i++) {
  //   weighted_in_[i-1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], 1);
  //   weights_[i-1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], this->params_.net_structure[i-1]);
  //   biases_[i-1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], 1);
  // }
}

// template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
// void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::updateModel(std::vector<int> description,
//        std::vector<float> data) {
//    // Updating only the latest state and control
//    if (description.size() == 2) {
//      if (description[0] != this->params_.DYNAMICS_DIM ||
//          description[1] != C_DIM) {
//        std::cerr << "Invalid update for LSTM Dyanmics. Expected "
//                  << this->params_.DYNAMICS_DIM << ", " << C_DIM << " and received"
//                  << description[0] << ", " << description[1] << std::endl;
//        exit(1);
//      }
//    } else { // Online uppdating of weights
//      this->params_.copy_everything = true; // Double check if this is needed?
//      // TODO
//    }
//  if(this->GPUMemStatus_) {
//    paramsToDevice();
//  }
//}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::paramsToDevice()
{
  // TODO copy to constant memory
  HANDLE_ERROR(cudaMemcpyAsync(this->model_d_->control_rngs_, this->control_rngs_, 2 * C_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));
  if (init_)
  {
    this->params_.updateInitialLSTMState();
  }
  if (this->params_.copy_everything)
  {
    // Copy Weight Matrices
    HANDLE_ERROR(cudaMemcpyAsync(&this->model_d_->params_, &this->params_, sizeof(this->params_),
                                 cudaMemcpyHostToDevice, this->stream_));
    this->params_.copy_everything = false;
  }
  else
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->model_d_->params_.initial_hidden, this->params_.initial_hidden,
                                 this->params_.HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->model_d_->params_.initial_cell, this->params_.initial_cell,
                                 this->params_.HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice, this->stream_));
  }

  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::loadParams(const std::string& lstm_model_path,
                                                                         const std::string& hidden_model_path,
                                                                         const std::string& cell_model_path,
                                                                         const std::string& output_model_path)
{
  std::string bias_name = "";
  std::string weight_name = "";
  if (!fileExists(lstm_model_path))
  {
    std::cerr << "Could not load neural net model at path: " << lstm_model_path.c_str();
    exit(-1);
  }
  if (!fileExists(hidden_model_path))
  {
    std::cerr << "Could not load neural net model at path: " << hidden_model_path.c_str();
    exit(-1);
  }
  if (!fileExists(cell_model_path))
  {
    std::cerr << "Could not load neural net model at path: " << cell_model_path.c_str();
    exit(-1);
  }
  if (!fileExists(output_model_path))
  {
    std::cerr << "Could not load neural net model at path: " << output_model_path.c_str();
    exit(-1);
  }
  // TODO: Read weights from npz file and load them into params
  cnpy::npz_t hidden_param_dict = cnpy::npz_load(hidden_model_path);
  cnpy::npz_t lstm_param_dict = cnpy::npz_load(lstm_model_path);
  cnpy::npz_t cell_param_dict = cnpy::npz_load(cell_model_path);
  cnpy::npz_t output_param_dict = cnpy::npz_load(output_model_path);

  /**
   * Load initialization network weights
   */
  cnpy::NpyArray hidden_weight_1_raw = hidden_param_dict["dynamics_W1"];
  double* hidden_weight_1 = hidden_weight_1_raw.data<double>();
  cnpy::NpyArray cell_weight_1_raw = cell_param_dict["dynamics_W1"];
  double* cell_weight_1 = cell_weight_1_raw.data<double>();
  for (int i = 0; i < this->params_.BUFFER_INTER_SIZE; i++)
  {
    this->params_.W_hidden_input.get()[i] = hidden_weight_1[i];
    this->params_.W_cell_input.get()[i] = cell_weight_1[i];
  }

  cnpy::NpyArray hidden_weight_2_raw = hidden_param_dict["dynamics_W2"];
  double* hidden_weight_2 = hidden_weight_2_raw.data<double>();
  cnpy::NpyArray cell_weight_2_raw = cell_param_dict["dynamics_W2"];
  double* cell_weight_2 = cell_weight_2_raw.data<double>();
  for (int i = 0; i < this->params_.INTER_HIDDEN_SIZE; i++)
  {
    this->params_.W_hidden_output.get()[i] = hidden_weight_2[i];
    this->params_.W_cell_output.get()[i] = cell_weight_2[i];
  }

  cnpy::NpyArray hidden_bias_1_raw = hidden_param_dict["dynamics_b1"];
  double* hidden_bias_1 = hidden_bias_1_raw.data<double>();
  cnpy::NpyArray cell_bias_1_raw = cell_param_dict["dynamics_b1"];
  double* cell_bias_1 = cell_bias_1_raw.data<double>();
  for (int i = 0; i < INIT_DIM; i++)
  {
    this->params_.b_hidden_input.get()[i] = hidden_bias_1[i];
    this->params_.b_cell_input.get()[i] = cell_bias_1[i];
  }

  cnpy::NpyArray hidden_bias_2_raw = hidden_param_dict["dynamics_b2"];
  double* hidden_bias_2 = hidden_bias_2_raw.data<double>();
  cnpy::NpyArray cell_bias_2_raw = cell_param_dict["dynamics_b2"];
  double* cell_bias_2 = cell_bias_2_raw.data<double>();
  for (int i = 0; i < this->params_.HIDDEN_DIM; i++)
  {
    this->params_.b_hidden_output.get()[i] = hidden_bias_2[i];
    this->params_.b_cell_output.get()[i] = cell_bias_2[i];
  }

  cnpy::NpyArray output_weight_raw = output_param_dict["dynamics_W1"];
  double* output_weight = output_weight_raw.data<double>();
  cnpy::NpyArray output_bias_raw = output_param_dict["dynamics_b1"];
  double* output_bias = output_bias_raw.data<double>();
  for (int i = 0; i < this->params_.STATE_HIDDEN_SIZE; i++)
  {
    this->params_.W_y[i] = output_weight[i];
    if (!isfinite(this->params_.W_y[i]))
    {
      std::cout << "setting W_y to 0" << std::endl;
      this->params_.W_y[i] = 0;
    }
    // std::cout << "output weight" << i << " " << this->params_.W_y[i] << " " << output_weight[i] << std::endl;
  }
  for (int i = 0; i < this->params_.DYNAMICS_DIM; i++)
  {
    this->params_.b_y[i] = output_bias[i];
  }

  /** LSTM Network Loading **/
  cnpy::NpyArray lstm_ih_weight_raw = lstm_param_dict["weight_ih_l0"];
  double* lstm_ih_weight = lstm_ih_weight_raw.data<double>();
  for (int i = 0; i < 4 * this->params_.STATE_HIDDEN_SIZE; i++)
  {
    if (i < this->params_.STATE_HIDDEN_SIZE)
    {
      // input gate
      this->params_.W_ii[i % this->params_.STATE_HIDDEN_SIZE] = lstm_ih_weight[i];
    }
    else if (i < 2 * this->params_.STATE_HIDDEN_SIZE)
    {
      // forget gate
      this->params_.W_fi[i % this->params_.STATE_HIDDEN_SIZE] = lstm_ih_weight[i];
    }
    else if (i < 3 * this->params_.STATE_HIDDEN_SIZE)
    {
      // cell gate
      this->params_.W_ci[i % this->params_.STATE_HIDDEN_SIZE] = lstm_ih_weight[i];
    }
    else if (i < 4 * this->params_.STATE_HIDDEN_SIZE)
    {
      // output gate
      this->params_.W_oi[i % this->params_.STATE_HIDDEN_SIZE] = lstm_ih_weight[i];
    }
  }

  cnpy::NpyArray lstm_hh_weight_raw = lstm_param_dict["weight_hh_l0"];
  double* lstm_hh_weight = lstm_hh_weight_raw.data<double>();
  for (int i = 0; i < 4 * this->params_.HIDDEN_HIDDEN_SIZE; i++)
  {
    if (i < this->params_.HIDDEN_HIDDEN_SIZE)
    {
      // input gate
      this->params_.W_im[i % this->params_.HIDDEN_HIDDEN_SIZE] = lstm_hh_weight[i];
    }
    else if (i < 2 * this->params_.HIDDEN_HIDDEN_SIZE)
    {
      // forget gate
      this->params_.W_fm[i % this->params_.HIDDEN_HIDDEN_SIZE] = lstm_hh_weight[i];
    }
    else if (i < 3 * this->params_.HIDDEN_HIDDEN_SIZE)
    {
      // cell gate
      this->params_.W_cm[i % this->params_.HIDDEN_HIDDEN_SIZE] = lstm_hh_weight[i];
    }
    else if (i < 4 * this->params_.HIDDEN_HIDDEN_SIZE)
    {
      // output gate
      this->params_.W_om[i % this->params_.HIDDEN_HIDDEN_SIZE] = lstm_hh_weight[i];
    }
  }

  cnpy::NpyArray lstm_hh_bias_raw = lstm_param_dict["bias_hh_l0"];
  double* lstm_hh_bias = lstm_hh_bias_raw.data<double>();
  cnpy::NpyArray lstm_ih_bias_raw = lstm_param_dict["bias_ih_l0"];
  double* lstm_ih_bias = lstm_ih_bias_raw.data<double>();
  for (int i = 0; i < 4 * this->params_.HIDDEN_DIM; i++)
  {
    if (i < this->params_.HIDDEN_DIM)
    {
      // input gate
      this->params_.b_i[i % this->params_.HIDDEN_DIM] = lstm_hh_bias[i] + lstm_ih_bias[i];
    }
    else if (i < 2 * this->params_.HIDDEN_DIM)
    {
      // forget gate
      this->params_.b_f[i % this->params_.HIDDEN_DIM] = lstm_hh_bias[i] + lstm_ih_bias[i];
    }
    else if (i < 3 * this->params_.HIDDEN_DIM)
    {
      // cell gate
      this->params_.b_c[i % this->params_.HIDDEN_DIM] = lstm_hh_bias[i] + lstm_ih_bias[i];
    }
    else if (i < 4 * this->params_.HIDDEN_DIM)
    {
      // output gate
      this->params_.b_o[i % this->params_.HIDDEN_DIM] = lstm_hh_bias[i] + lstm_ih_bias[i];
    }
  }

  // TODO ensure all finite
  std::cout << "loaded NN" << std::endl;
  for (int i = 0; i < this->params_.BUFFER_INTER_SIZE; i++)
  {
    assert(isfinite(this->params_.W_hidden_input.get()[i]));
    assert(isfinite(this->params_.W_cell_input.get()[i]));
  }
  for (int i = 0; i < this->params_.INTER_HIDDEN_SIZE; i++)
  {
    assert(isfinite(this->params_.W_hidden_output.get()[i]));
    assert(isfinite(this->params_.W_cell_output.get()[i]));
  }
  for (int i = 0; i < INIT_DIM; i++)
  {
    assert(isfinite(this->params_.b_hidden_input.get()[i]));
    assert(isfinite(this->params_.b_cell_input.get()[i]));
  }
  for (int i = 0; i < this->params_.HIDDEN_DIM; i++)
  {
    assert(isfinite(this->params_.b_hidden_output.get()[i]));
    assert(isfinite(this->params_.b_cell_output.get()[i]));
  }
  for (int i = 0; i < this->params_.STATE_HIDDEN_SIZE; i++)
  {
    assert(isfinite(this->params_.W_y[i]));
  }
  for (int i = 0; i < this->params_.DYNAMICS_DIM; i++)
  {
    assert(isfinite(this->params_.b_y[i]));
  }
  for (int i = 0; i < this->params_.STATE_HIDDEN_SIZE; i++)
  {
    assert(isfinite(this->params_.W_ii[i]));
    assert(isfinite(this->params_.W_fi[i]));
    assert(isfinite(this->params_.W_ci[i]));
    assert(isfinite(this->params_.W_oi[i]));
  }
  for (int i = 0; i < this->params_.HIDDEN_HIDDEN_SIZE; i++)
  {
    assert(isfinite(this->params_.W_im[i]));
    assert(isfinite(this->params_.W_fm[i]));
    assert(isfinite(this->params_.W_cm[i]));
    assert(isfinite(this->params_.W_om[i]));
  }
  for (int i = 0; i < this->params_.HIDDEN_DIM; i++)
  {
    assert(isfinite(this->params_.b_i[i]));
    assert(isfinite(this->params_.b_f[i]));
    assert(isfinite(this->params_.b_c[i]));
    assert(isfinite(this->params_.b_o[i]));
  }

  this->init_ = true;
  // Save parameters to GPU memory
  this->params_.copy_everything = true;
  if (this->GPUMemStatus_)
  {
    paramsToDevice();
  }
}

// template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
// bool LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::computeGrad(const Eigen::Ref<const state_array>& state,
//                                                                      const Eigen::Ref<const control_array>& control,
//                                                                      Eigen::Ref<dfdx> A,
//                                                                      Eigen::Ref<dfdu> B) {
//   // TODO results are not returned
//   Eigen::Matrix<float, S_DIM, S_DIM + C_DIM> jac;
//   jac.setZero();

//   //Start with the kinematic and physics model derivatives
//   jac.row(0) << 0, 0, -sin(state(2))*state(4) - cos(state(2))*state(5), 0, cos(state(2)), -sin(state(2)), 0, 0, 0;
//   jac.row(1) << 0, 0, cos(state(2))*state(4) - sin(state(2))*state(5), 0, sin(state(2)), cos(state(2)), 0, 0, 0;
//   jac.row(2) << 0, 0, 0, 0, 0, 0, -1, 0, 0;

//   state_array state_der;

//   //First do the forward pass
//   computeDynamics(state, control, state_der);

//   //Start backprop
//   Eigen::MatrixXf ip_delta = Eigen::MatrixXf::Identity(DYNAMICS_DIM, DYNAMICS_DIM);
//   Eigen::MatrixXf temp_delta = Eigen::MatrixXf::Identity(DYNAMICS_DIM, DYNAMICS_DIM);

//   //Main backprop loop
//   for (int i = NUM_LAYERS-2; i > 0; i--){
//     Eigen::MatrixXf zp = weighted_in_[i-1];
//     for (int j = 0; j < this->params_.net_structure[i]; j++){
//       zp(j) = MPPI_NNET_NONLINEARITY_DERIV(zp(j));
//     }
//     ip_delta =  ( (weights_[i]).transpose()*ip_delta).eval();
//     for (int j = 0; j < DYNAMICS_DIM; j++){
//       ip_delta.col(j) = ip_delta.col(j).array() * zp.array();
//     }
//   }
//   //Finish the backprop loop
//   ip_delta = ( ((weights_[0]).transpose())*ip_delta).eval();
//   jac.bottomRightCorner(DYNAMICS_DIM, DYNAMICS_DIM + C_DIM) += ip_delta.transpose();
//   A = jac.leftCols(S_DIM);
//   B = jac.rightCols(C_DIM);
//   return true;
// }

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::computeKinematics(
    const Eigen::Ref<const state_array>& state, Eigen::Ref<state_array> state_der)
{
  state_der(0) = cosf(state(2)) * state(4) - sinf(state(2)) * state(5);
  state_der(1) = sinf(state(2)) * state(4) + cosf(state(2)) * state(5);
  state_der(2) = -state(6);  // Pose estimate actually gives the negative yaw derivative
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::computeDynamics(
    const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    Eigen::Ref<state_array> state_der)
{
  // TODO: Figure out LSTM on CPU
  using W_hh = Eigen::Matrix<float, H_DIM, H_DIM, Eigen::RowMajor>;
  using W_hi = Eigen::Matrix<float, H_DIM, C_DIM + DYNAMICS_DIM, Eigen::RowMajor>;
  using W_ih = Eigen::Matrix<float, C_DIM + DYNAMICS_DIM, H_DIM, Eigen::RowMajor>;
  using b_output = Eigen::Matrix<float, C_DIM + DYNAMICS_DIM, 1>;
  using output_layer = Eigen::Matrix<float, H_DIM, 1>;

  // Create eigen matrices for all the weights and biases
  Eigen::Map<const W_hh> W_im_mat(this->params_.W_im);
  Eigen::Map<const W_hh> W_fm_mat(this->params_.W_fm);
  Eigen::Map<const W_hh> W_om_mat(this->params_.W_om);
  Eigen::Map<const W_hh> W_cm_mat(this->params_.W_cm);

  Eigen::Map<const W_hi> W_ii_mat(this->params_.W_ii);
  Eigen::Map<const W_hi> W_fi_mat(this->params_.W_fi);
  Eigen::Map<const W_hi> W_oi_mat(this->params_.W_oi);
  Eigen::Map<const W_hi> W_ci_mat(this->params_.W_ci);
  Eigen::Map<const W_ih> W_y_mat(this->params_.W_y);

  Eigen::Map<const output_layer> b_i_mat(this->params_.b_i);
  Eigen::Map<const output_layer> b_f_mat(this->params_.b_f);
  Eigen::Map<const output_layer> b_o_mat(this->params_.b_o);
  Eigen::Map<const output_layer> b_c_mat(this->params_.b_c);
  Eigen::Map<const b_output> b_y_mat(this->params_.b_y);

  b_output input_mat;
  input_mat << state[3], state[4], state[5], state[6], control[0], control[1];
  output_layer g_i = W_im_mat * hidden_state_ + W_ii_mat * input_mat + b_i_mat;
  output_layer g_f = W_fm_mat * hidden_state_ + W_fi_mat * input_mat + b_f_mat;
  output_layer g_o = W_om_mat * hidden_state_ + W_oi_mat * input_mat + b_o_mat;
  output_layer g_c = W_cm_mat * hidden_state_ + W_ci_mat * input_mat + b_c_mat;
  g_i = g_i.unaryExpr([](float x) { return 1.0f / (1 + expf(-x)); });
  g_f = g_f.unaryExpr([](float x) { return 1.0f / (1 + expf(-x)); });
  g_o = g_o.unaryExpr([](float x) { return 1.0f / (1 + expf(-x)); });
  g_c = g_c.unaryExpr([](float x) { return tanhf(x); });

  output_layer c_next = g_i.cwiseProduct(g_c) + g_f.cwiseProduct(cell_state_);
  output_layer h_next = g_o.cwiseProduct(c_next.unaryExpr([](float x) { return tanhf(x); }));
  b_output state_der_small;
  state_der_small = W_y_mat * h_next + b_y_mat;
  for (int i = 0; i < DYNAMICS_DIM; i++)
  {
    state_der[i + (S_DIM - DYNAMICS_DIM)] = state_der_small[i];
  }
  // state_der[6] *= -1;
  hidden_state_ = h_next;
  cell_state_ = c_next;
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
__device__ void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::computeKinematics(float* state,
                                                                                           float* state_der)
{
  state_der[0] = cosf(state[2]) * state[4] - sinf(state[2]) * state[5];
  state_der[1] = sinf(state[2]) * state[4] + cosf(state[2]) * state[5];
  state_der[2] = -state[6];  // Pose estimate actually gives the negative yaw derivative
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
__device__ void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::initializeDynamics(
    float* state, float* control, float* output, float* theta_s, float t_0, float dt)
{
  int shared_mem_size = this->params_.SHARED_MEM_REQUEST_BLK_BYTES;
  int block_idx = (blockDim.x * threadIdx.z + threadIdx.x) * (shared_mem_size);

  float* c = &theta_s[block_idx];
  float* h = &theta_s[block_idx + H_DIM];
  for (int i = threadIdx.y; i < PARENT_CLASS::OUTPUT_DIM && i < PARENT_CLASS::STATE_DIM; ++i)
  {
    output[i] = state[i];
  }
  for (int i = threadIdx.y; i < H_DIM; i += blockDim.y)
  {
    c[i] = this->params_.initial_cell[i];
    h[i] = this->params_.initial_hidden[i];
  }
  __syncthreads();
}

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::initializeDynamics(
    const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    Eigen::Ref<output_array> output, float t_0, float dt)
{
  Eigen::Map<Eigen::Matrix<float, H_DIM, 1>> init_hidden(this->params_.initial_hidden);
  Eigen::Map<Eigen::Matrix<float, H_DIM, 1>> init_cell(this->params_.initial_cell);
  output = state;
  hidden_state_ = init_hidden;
  cell_state_ = init_cell;
}

// x = v_k
// h = m_{k-1}
// c = c_{k-1}
template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
__device__ void LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::computeDynamics(float* state, float* control,
                                                                                         float* state_der,
                                                                                         float* theta_s)
{
  // float* curr_act;
  // float* next_act;
  // float* tmp_act;
  int tdy = threadIdx.y;
  // Weights
  float* W_ii = (float*)&(this->params_.W_ii);
  float* W_im = (float*)&(this->params_.W_im);
  float* W_fi = (float*)&(this->params_.W_fi);
  float* W_fm = (float*)&(this->params_.W_fm);
  float* W_oi = (float*)&(this->params_.W_oi);
  float* W_om = (float*)&(this->params_.W_om);
  float* W_ci = (float*)&(this->params_.W_ci);
  float* W_cm = (float*)&(this->params_.W_cm);

  // float* W_im ; // hidden * state, hidden * hidden
  // float* W_fi, *W_fm; // hidden * state, hidden * hidden
  // float* W_oi, *W_om; // hidden * state, hidden * hidden
  // float* W_ci, *W_cm; // hidden * state, hidden * hidden
  // Biases
  float* b_i = (float*)&(this->params_.b_i);  // hidden_size
  float* b_f = (float*)&(this->params_.b_f);  // hidden_size
  float* b_o = (float*)&(this->params_.b_o);  // hidden_size
  float* b_c = (float*)&(this->params_.b_c);  // hidden_size
  // Intermediate outputs
  int shared_mem_size = this->params_.SHARED_MEM_REQUEST_BLK_BYTES;
  int block_idx = (blockDim.x * threadIdx.z + threadIdx.x) * (shared_mem_size);
  float* c = &theta_s[block_idx];
  float* h = &theta_s[block_idx + H_DIM];
  float* next_cell_state = &theta_s[block_idx + 2 * H_DIM];
  float* next_hidden_state = &theta_s[block_idx + 3 * H_DIM];
  float* g_i = &theta_s[block_idx + 4 * H_DIM];  // input gate
  float* g_f = &theta_s[block_idx + 5 * H_DIM];  // forget gate
  float* g_o = &theta_s[block_idx + 6 * H_DIM];  // output gate
  float* cell_update = &theta_s[block_idx + 7 * H_DIM];
  float* x = &theta_s[block_idx + 8 * H_DIM];

  // float* g_i, *g_f, *g_o, *cell_update; // hidden_size

  float* W_y = (float*)&(this->params_.W_y);  // state * hidden
  float* b_y = (float*)&(this->params_.b_y);  // state
  int i, j;
  int input_size = DYNAMICS_DIM + C_DIM;
  int hidden_size = H_DIM;

  // float* intermediate_y;
  int index = 0;

  // iterate through the part of the state that should be an input to the NN
  for (i = tdy; i < DYNAMICS_DIM; i += blockDim.y)
  {
    x[i] = state[i + (S_DIM - DYNAMICS_DIM)];
  }
  // iterate through the control to put into first layer
  for (i = tdy; i < C_DIM; i += blockDim.y)
  {
    x[DYNAMICS_DIM + i] = control[i];
  }
  __syncthreads();
  // input gate
  // for (i = 0; i < hidden_size; i++) {
  //   g_i[i] = 0;
  //   for (j = 0; j < input_size; j++) {
  //     g_i[i] += W_ii[i * input_size + j] * x[j];
  //   }
  //   for (j = 0; j < hidden_size; j++) {
  //     index = i * hidden_size + j;
  //     g_i[i] += W_im[index] * h[j];
  //   }
  //   g_i[i] += b_i[i];
  //   g_i[i] = SIGMOID(g_i[i]);
  // }
  // // forget gate
  // for (i = 0; i < hidden_size; i++) {
  //   g_f[i] = 0;
  //   for (j = 0; j < input_size; j++) {
  //     g_f[i] += W_fi[i * input_size + j] * x[j];
  //   }
  //   for (j = 0; j < hidden_size; j++) {
  //     index = i * hidden_size + j;
  //     g_f[i] += W_fm[index] * h[j];
  //   }
  //   g_f[i] += b_f[i];
  //   g_f[i] = SIGMOID(g_f[i]);
  // }
  // // output gate
  // for (i = 0; i < hidden_size; i++) {
  //   g_o[i] = 0;
  //   for (j = 0; j < input_size; j++) {
  //     g_o[i] += W_oi[i * input_size + j] * x[j];
  //   }
  //   for (j = 0; j < hidden_size; j++) {
  //     index = i * hidden_size + j;
  //     g_o[i] += W_om[index] * h[j];
  //   }
  //   g_o[i] += b_o[i];
  //   g_o[i] = SIGMOID(g_o[i]);
  // }
  // // cell update
  // for (i = 0; i < hidden_size; i++) {
  //   cell_update[i] = 0;
  //   for (j = 0; j < input_size; j++) {
  //     cell_update[i] += W_ci[i * input_size + j] * x[j];
  //   }
  //   for (j = 0; j < hidden_size; j++) {
  //     cell_update[i] += W_cm[i * hidden_size + j] * h[j];
  //   }
  //   cell_update[i] += b_c[i];
  //   cell_update[i] = RELU(cell_update[i]);
  // }

  // Update gates in parallel
  for (i = tdy; i < hidden_size; i += blockDim.y)
  {
    g_i[i] = 0;
    g_f[i] = 0;
    g_o[i] = 0;
    cell_update[i] = 0;
    for (j = 0; j < input_size; j++)
    {
      index = i * input_size + j;
      g_i[i] += W_ii[index] * x[j];
      g_f[i] += W_fi[index] * x[j];
      g_o[i] += W_oi[index] * x[j];
      cell_update[i] += W_ci[index] * x[j];
    }
    for (j = 0; j < hidden_size; j++)
    {
      index = i * hidden_size + j;
      g_i[i] += W_im[index] * h[j];
      g_f[i] += W_fm[index] * h[j];
      g_o[i] += W_om[index] * h[j];
      cell_update[i] += W_cm[index] * h[j];
    }
    g_i[i] += b_i[i];
    g_f[i] += b_f[i];
    g_o[i] += b_o[i];
    cell_update[i] += b_c[i];
    g_i[i] = SIGMOID(g_i[i]);
    g_f[i] = SIGMOID(g_f[i]);
    g_o[i] = SIGMOID(g_o[i]);
    cell_update[i] = tanhf(cell_update[i]);
  }
  __syncthreads();
  // outputs in parallel
  for (i = tdy; i < hidden_size; i += blockDim.y)
  {
    next_cell_state[i] = g_i[i] * cell_update[i] + g_f[i] * c[i];
    next_hidden_state[i] = tanhf(next_cell_state[i]) * g_o[i];
  }
  __syncthreads();

  // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
  //   printf("W_ii[6]: %f\n", W_ii[6]);
  //   printf("Initial state: ");
  //   for (i = 0; i < input_size; i++) {
  //     printf("%f, ", x[i]);
  //   }
  //   printf("\n");
  //   printf("Hidden state: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", h[i]);
  //   }
  //   printf("\n");

  //   printf("Cell state: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", c[i]);
  //   }
  //   printf("\n");
  //   printf("\033[1;32mIn gate after sigmoid: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", g_i[i]);
  //   }
  //   printf("\033[0m\n");
  //   printf("\033[1;33mForget gate after sigmoid: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", g_f[i]);
  //   }
  //   printf("\033[0m\n");
  //   printf("\033[1;34mOut gate after sigmoid: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", g_o[i]);
  //   }
  //   printf("\033[0m\n");
  //   printf("\033[1;35mCell gate after tanh: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", cell_update[i]);
  //   }
  //   printf("\033[0m\n");
  //   printf("Next hidden state: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", next_hidden_state[i]);
  //   }
  //   printf("\n");
  //   printf("Next cell state: ");
  //   for (i = 0; i < hidden_size; i++) {
  //     printf("%f, ", next_cell_state[i]);
  //   }
  //   printf("\n");
  // }

  // __syncthreads();
  // TODO: Copy next hidden/cell state into previous hidden/cell
  for (i = tdy; i < hidden_size; i += blockDim.y)
  {
    c[i] = next_cell_state[i];
    h[i] = next_hidden_state[i];
  }

  for (i = tdy; i < DYNAMICS_DIM; i += blockDim.y)
  {
    state_der[i + (S_DIM - DYNAMICS_DIM)] = 0;
    for (j = 0; j < hidden_size; j++)
    {
      state_der[i + (S_DIM - DYNAMICS_DIM)] += W_y[i * hidden_size + j] * next_hidden_state[j];
    }
    state_der[i + (S_DIM - DYNAMICS_DIM)] += b_y[i];
  }
  // outputs
  // for (i = 0; i < hidden_size; i++) {
  //   next_cell_state[i] = g_i[i] * cell_update[i] + g_f[i] * c[j];
  // }
  // for (i = 0; i < hidden_size; i++) {
  //   next_hidden_state[i] = tanhf(next_cell_state[i]) * g_o[i];
  // }

  // for (i = 0; i < input_size; i++) {
  //   intermediate_y = 0;
  //   for (j = 0; j < hidden_size; j++) {
  //     intermediate_y += W_y[i * hidden_size + j] * next_hidden_state[j];
  //   }
  //   intermediate_y += b_y[i];
  //   intermediate_y[i] = intermediate_y;
  //   state_der[i] = x[i] + intermediate_y[i] * this->params_.dt;
  // }
  __syncthreads();
  // if (threadIdx.y == 0) {
  //  state_der[6] *= -1;
  //}
  //__syncthreads();
}
