//
// Created by jason on 8/20/22.
//

#include "lstm_helper.cuh"

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::LSTMHelper(cudaStream_t stream) : Managed(stream)
{
  output_nn_ = new OUTPUT_FNN_T(stream);
  hidden_state_ = hidden_state::Zero();
  cell_state_ = hidden_state::Zero();
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::LSTMHelper(std::string path, cudaStream_t stream) : Managed(stream)
{
  output_nn_ = new OUTPUT_FNN_T(stream);
  hidden_state_ = hidden_state::Zero();
  cell_state_ = hidden_state::Zero();
  loadParams(path);
}
template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::GPUSetup()
{
  output_nn_->GPUSetup();
  if (!this->GPUMemStatus_)
  {
    network_d_ = Managed::GPUSetup<LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>>(this);
    HANDLE_ERROR(cudaMemcpyAsync(&(this->network_d_->output_nn_), &(this->output_nn_->network_d_),
                                 sizeof(OUTPUT_FNN_T*), cudaMemcpyHostToDevice, this->stream_));
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;
  }
}
template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::freeCudaMem()
{
  output_nn_->freeCudaMem();
  if (this->GPUMemStatus_)
  {
    cudaFree(network_d_);
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    // copies entire params to device
    HANDLE_ERROR(
        cudaMemcpyAsync(&this->network_d_->params_, &this->params_, sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
__device__ void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::initialize(float* theta_s)
{
  static_assert(std::is_trivially_copyable<PARAMS_T>::value);
  const int slide = LSTM_SHARED_MEM_GRD / sizeof(float) + 1;

  PARAMS_T* lstm_params = &this->params_;
  OUTPUT_PARAMS_T* output_params = this->output_nn_->getParamsPtr();
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    lstm_params = (PARAMS_T*)theta_s;
    output_params = (OUTPUT_PARAMS_T*)(theta_s + slide);
  }

  const int block_idx = (blockDim.x * threadIdx.z + threadIdx.x) * SHARED_MEM_REQUEST_BLK_BYTES / sizeof(float) +
                        SHARED_MEM_REQUEST_GRD_BYTES / sizeof(float) + 1;

  initialize(lstm_params, output_params, theta_s + block_idx);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
__device__ void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::initialize(LSTM_PARAMS_T* lstm_params,
                                                                           OUTPUT_PARAMS_T* output_params,
                                                                           float* hidden_cell)
{
  // if using shared memory, copy parameters
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    *lstm_params = this->params_;
    output_nn_->initialize(output_params);
  }

  float* c = hidden_cell;
  float* h = hidden_cell + HIDDEN_DIM;

  if (SHARED_MEM_REQUEST_GRD_BYTES == 0)
  {
    for (int i = threadIdx.y; i < HIDDEN_DIM; i += blockDim.y)
    {
      c[i] = params_.initial_cell[i];
      h[i] = params_.initial_hidden[i];
    }
  }
  else
  {
    for (int i = threadIdx.y; i < HIDDEN_DIM; i += blockDim.y)
    {
      c[i] = lstm_params->initial_cell[i];
      h[i] = lstm_params->initial_hidden[i];
    }
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::copyHiddenCellToDevice()
{
  if (this->GPUMemStatus_)
  {
    HANDLE_ERROR(cudaMemcpyAsync(&this->network_d_->params_.initial_hidden, &this->params_.initial_hidden,
                                 HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice, stream_));
    HANDLE_ERROR(cudaMemcpyAsync(&this->network_d_->params_.initial_cell, &this->params_.initial_cell,
                                 HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice, stream_));
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::updateOutputModel(const std::vector<int>& description,
                                                                       const std::vector<float>& data)
{
  output_nn_->updateModel(description, data);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::setLSTMParams(PARAMS_T& params)
{
  params_ = params;
  resetHiddenCellCPU();
  paramsToDevice();
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::forward(const Eigen::Ref<const input_array>& input)
{
  // Create eigen matrices for all the weights and biases
  Eigen::Map<const W_hh> W_im_mat(this->params_.W_im);
  Eigen::Map<const W_hh> W_fm_mat(this->params_.W_fm);
  Eigen::Map<const W_hh> W_om_mat(this->params_.W_om);
  Eigen::Map<const W_hh> W_cm_mat(this->params_.W_cm);

  Eigen::Map<const W_hi> W_ii_mat(this->params_.W_ii);
  Eigen::Map<const W_hi> W_fi_mat(this->params_.W_fi);
  Eigen::Map<const W_hi> W_oi_mat(this->params_.W_oi);
  Eigen::Map<const W_hi> W_ci_mat(this->params_.W_ci);

  Eigen::Map<const hidden_state> b_i_mat(this->params_.b_i);
  Eigen::Map<const hidden_state> b_f_mat(this->params_.b_f);
  Eigen::Map<const hidden_state> b_o_mat(this->params_.b_o);
  Eigen::Map<const hidden_state> b_c_mat(this->params_.b_c);

  hidden_state g_i = W_im_mat * hidden_state_ + W_ii_mat * input + b_i_mat;
  hidden_state g_f = W_fm_mat * hidden_state_ + W_fi_mat * input + b_f_mat;
  hidden_state g_o = W_om_mat * hidden_state_ + W_oi_mat * input + b_o_mat;
  hidden_state g_c = W_cm_mat * hidden_state_ + W_ci_mat * input + b_c_mat;
  g_i = g_i.unaryExpr([](float x) { return SIGMOID(x); });
  g_f = g_f.unaryExpr([](float x) { return SIGMOID(x); });
  g_o = g_o.unaryExpr([](float x) { return SIGMOID(x); });
  g_c = g_c.unaryExpr([](float x) { return TANH(x); });

  hidden_state c_next = g_i.cwiseProduct(g_c) + g_f.cwiseProduct(cell_state_);
  hidden_state h_next = g_o.cwiseProduct(c_next.unaryExpr([](float x) { return tanhf(x); }));

  hidden_state_ = h_next;
  cell_state_ = c_next;
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::forward(const Eigen::Ref<const input_array>& input,
                                                             Eigen::Ref<output_array> output)
{
  forward(input);
  typename OUTPUT_FNN_T::input_array nn_input;
  nn_input.head(HIDDEN_DIM) = hidden_state_;
  nn_input.tail(INPUT_DIM) = input;

  output_nn_->forward(nn_input, output);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
__device__ float* LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::forward(float* input, float* theta_s)
{
  PARAMS_T* params = &this->params_;
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    params = (PARAMS_T*)theta_s;
  }

  const int block_idx = (blockDim.x * threadIdx.z + threadIdx.x) * SHARED_MEM_REQUEST_BLK_BYTES / sizeof(float) +
                        SHARED_MEM_REQUEST_GRD_BYTES / sizeof(float) + 1;
  return forward(input, theta_s, params, block_idx);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
__device__ float* LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::forward(float* input, float* theta_s,
                                                                          LSTM_PARAMS_T* params, int block_idx)
{
  FNN_PARAMS_T* output_params;
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    const int slide = LSTM_SHARED_MEM_GRD / sizeof(float) + 1;
    output_params = (FNN_PARAMS_T*)(theta_s + slide);
  }
  else
  {
    output_params = output_nn_->getParamsPtr();
  }

  return forward(input, theta_s, params, output_params, block_idx);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
__device__ float* LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::forward(float* input, float* theta_s,
                                                                          LSTM_PARAMS_T* params,
                                                                          FNN_PARAMS_T* output_params, int block_idx)
{
  float* c = &theta_s[block_idx];
  float* g_o = &theta_s[block_idx + 2 * HIDDEN_DIM];  // input gate
  return forward(input, g_o, c, params, output_params, 0);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
__device__ float* LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::forward(float* input, float* theta_s,
                                                                          float* hidden_cell, LSTM_PARAMS_T* params,
                                                                          FNN_PARAMS_T* output_params, int block_idx)
{
  // Weights
  const float* W_ii = &params->W_ii[0];
  const float* W_im = &params->W_im[0];
  const float* W_fi = &params->W_fi[0];
  const float* W_fm = &params->W_fm[0];
  const float* W_oi = &params->W_oi[0];
  const float* W_om = &params->W_om[0];
  const float* W_ci = &params->W_ci[0];
  const float* W_cm = &params->W_cm[0];

  // Biases
  const float* b_i = &params->b_i[0];  // hidden_size
  const float* b_f = &params->b_f[0];  // hidden_size
  const float* b_o = &params->b_o[0];  // hidden_size
  const float* b_c = &params->b_c[0];  // hidden_size

  uint i, j;

  // Intermediate outputs
  float* c = &hidden_cell[block_idx];
  float* h = &hidden_cell[block_idx + HIDDEN_DIM];
  float* g_o = &theta_s[block_idx];  // output gate
  float* x = &theta_s[block_idx + HIDDEN_DIM];
  float* output_act = &theta_s[block_idx + HIDDEN_DIM + INPUT_DIM];

  uint tdy = threadIdx.y;

  // load input into theta_s
  if (input != nullptr)
  {
    for (i = tdy; i < INPUT_DIM; i += blockDim.y)
    {
      x[i] = input[i];
    }
  }
  __syncthreads();

  int index = 0;
  // apply each gate in parallel

  float temp_g_i = 0;
  float temp_g_f = 0;
  float temp_g_o = 0;
  float temp_cell_update = 0;

  for (i = tdy; i < HIDDEN_DIM; i += blockDim.y)
  {
    temp_g_i = 0;
    temp_g_f = 0;
    temp_g_o = 0;
    temp_cell_update = 0;
    for (j = 0; j < INPUT_DIM; j++)
    {
      index = i * INPUT_DIM + j;
      temp_g_i += W_ii[index] * x[j];
      temp_g_f += W_fi[index] * x[j];
      temp_g_o += W_oi[index] * x[j];
      temp_cell_update += W_ci[index] * x[j];
    }
    for (j = 0; j < HIDDEN_DIM; j++)
    {
      index = i * HIDDEN_DIM + j;
      temp_g_i += W_im[index] * h[j];
      temp_g_f += W_fm[index] * h[j];
      temp_g_o += W_om[index] * h[j];
      temp_cell_update += W_cm[index] * h[j];
    }
    temp_g_i += b_i[i];
    temp_g_f += b_f[i];
    temp_g_o += b_o[i];
    temp_cell_update += b_c[i];

    temp_g_i = SIGMOID(temp_g_i);
    temp_g_f = SIGMOID(temp_g_f);
    temp_cell_update = TANH(temp_cell_update);

    g_o[i] = SIGMOID(temp_g_o);

    c[i] = temp_g_i * temp_cell_update + temp_g_f * c[i];
  }
  __syncthreads();

  // copy computed hidden/cell state to theta_s
  for (i = tdy; i < HIDDEN_DIM; i += blockDim.y)
  {
    h[i] = TANH(c[i]) * g_o[i];  // actually using c_next intentionally
    output_act[i] = h[i];
  }

  // copy input to activation
  for (i = tdy; i < INPUT_DIM; i += blockDim.y)
  {
    output_act[i + HIDDEN_DIM] = x[i];
  }

  return output_nn_->forward(nullptr, output_act, output_params, 0);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::resetHiddenCPU()
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    hidden_state_[i] = params_.initial_hidden[i];
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::resetCellCPU()
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    cell_state_[i] = params_.initial_cell[i];
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::resetHiddenCellCPU()
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    hidden_state_[i] = params_.initial_hidden[i];
    cell_state_[i] = params_.initial_cell[i];
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::setHiddenState(const Eigen::Ref<const hidden_state> hidden_state)
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    params_.initial_hidden[i] = hidden_state[i];
    hidden_state_[i] = hidden_state[i];
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::setCellState(const Eigen::Ref<const hidden_state> cell_state)
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    params_.initial_cell[i] = cell_state[i];
    cell_state_[i] = cell_state[i];
  }
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::loadParams(const std::string& model_path)
{
  if (!fileExists(model_path))
  {
    std::cerr << "Could not load neural net model at path: " << model_path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(model_path);
  loadParams(param_dict);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::loadParams(const cnpy::npz_t& param_dict)
{
  loadParams("", param_dict);
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
void LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::loadParams(std::string prefix, const cnpy::npz_t& param_dict,
                                                                bool add_slash)
{
  if (add_slash && !prefix.empty() && *prefix.rbegin() != '/')
  {
    prefix.append("/");
  }

  // assumes it has been unonioned
  output_nn_->loadParams(prefix + "output/", param_dict);

  cnpy::NpyArray weight_hh_raw = param_dict.at(prefix + "lstm/weight_hh_l0");
  cnpy::NpyArray bias_hh_raw = param_dict.at(prefix + "lstm/bias_hh_l0");
  cnpy::NpyArray weight_ih_raw = param_dict.at(prefix + "lstm/weight_ih_l0");
  cnpy::NpyArray bias_ih_raw = param_dict.at(prefix + "lstm/bias_ih_l0");
  double* weight_hh = weight_hh_raw.data<double>();
  double* bias_hh = bias_hh_raw.data<double>();
  double* weight_ih = weight_ih_raw.data<double>();
  double* bias_ih = bias_ih_raw.data<double>();

  for (int i = 0; i < PARAMS_T::HIDDEN_HIDDEN_SIZE; i++)
  {
    params_.W_im[i] = weight_hh[i];
    params_.W_fm[i] = weight_hh[i + PARAMS_T::HIDDEN_HIDDEN_SIZE];
    params_.W_cm[i] = weight_hh[i + 2 * PARAMS_T::HIDDEN_HIDDEN_SIZE];
    params_.W_om[i] = weight_hh[i + 3 * PARAMS_T::HIDDEN_HIDDEN_SIZE];
    assert(isfinite(params_.W_im[i]));
    assert(isfinite(params_.W_fm[i]));
    assert(isfinite(params_.W_cm[i]));
    assert(isfinite(params_.W_om[i]));
  }
  for (int i = 0; i < PARAMS_T::INPUT_HIDDEN_SIZE; i++)
  {
    params_.W_ii[i] = weight_ih[i];
    params_.W_fi[i] = weight_ih[i + PARAMS_T::INPUT_HIDDEN_SIZE];
    params_.W_ci[i] = weight_ih[i + 2 * PARAMS_T::INPUT_HIDDEN_SIZE];
    params_.W_oi[i] = weight_ih[i + 3 * PARAMS_T::INPUT_HIDDEN_SIZE];
    assert(isfinite(params_.W_ii[i]));
    assert(isfinite(params_.W_fi[i]));
    assert(isfinite(params_.W_ci[i]));
    assert(isfinite(params_.W_oi[i]));
  }
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    params_.b_i[i] = bias_hh[i] + bias_ih[i];
    params_.b_f[i] = bias_hh[i + HIDDEN_DIM] + bias_ih[i + HIDDEN_DIM];
    params_.b_c[i] = bias_hh[i + 2 * HIDDEN_DIM] + bias_ih[i + 2 * HIDDEN_DIM];
    params_.b_o[i] = bias_hh[i + 3 * HIDDEN_DIM] + bias_ih[i + 3 * HIDDEN_DIM];
    assert(isfinite(params_.b_i[i]));
    assert(isfinite(params_.b_f[i]));
    assert(isfinite(params_.b_c[i]));
    assert(isfinite(params_.b_o[i]));
  }

  // Save parameters to GPU memory
  paramsToDevice();
}

template <class PARAMS_T, class FNN_PARAMS_T, bool USE_SHARED>
__device__ float* LSTMHelper<PARAMS_T, FNN_PARAMS_T, USE_SHARED>::getInputLocation(float* theta_s)
{
  const int block_idx = (blockDim.x * threadIdx.z + threadIdx.x) * SHARED_MEM_REQUEST_BLK_BYTES / sizeof(float) +
                        SHARED_MEM_REQUEST_GRD_BYTES / sizeof(float) + 1;
  float* x = theta_s + block_idx + 3 * HIDDEN_DIM;
  return x;
}
