//
// Created by jason on 8/20/22.
//

#include "lstm_helper.cuh"

template <class PARAMS_T, class OUTPUT_FNN_T>
LSTMHelper<PARAMS_T, OUTPUT_FNN_T>::LSTMHelper<PARAMS_T, OUTPUT_FNN_T>(cudaStream_t stream) : Managed(stream)
{
  output_nn_ = new OUTPUT_FNN_T(stream);
  hidden_state_ = hidden_state::Zero();
  cell_state_ = hidden_state::Zero();
}

template <class PARAMS_T, class OUTPUT_FNN_T>
LSTMHelper<PARAMS_T, OUTPUT_FNN_T>::LSTMHelper<PARAMS_T, OUTPUT_FNN_T>(std::string path, cudaStream_t stream)
  : Managed(stream)
{
  output_nn_ = new OUTPUT_FNN_T(stream);
  hidden_state_ = hidden_state::Zero();
  cell_state_ = hidden_state::Zero();
  loadParams(path);
}
template <class PARAMS_T, class OUTPUT_FNN_T>
void LSTMHelper<PARAMS_T, OUTPUT_FNN_T>::GPUSetup()
{
  output_nn_->GPUSetup();
  if (!this->GPUMemStatus_)
  {
    network_d_ = Managed::GPUSetup<LSTMHelper<PARAMS_T, OUTPUT_FNN_T>>(this);
    HANDLE_ERROR(cudaMemcpyAsync(&(this->network_d_->output_nn_), &(this->output_nn_->network_d_),
                                 sizeof(OUTPUT_FNN_T*), cudaMemcpyHostToDevice, this->stream_));
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;
  }
}
template <class PARAMS_T, class OUTPUT_FNN_T>
void LSTMHelper<PARAMS_T, OUTPUT_FNN_T>::freeCudaMem()
{
  output_nn_->freeCudaMem();
  if (this->GPUMemStatus_)
  {
    cudaFree(network_d_);
  }
}

template <class PARAMS_T, class OUTPUT_FNN_T>
void LSTMHelper<PARAMS_T, OUTPUT_FNN_T>::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    // copies entire params to device
    HANDLE_ERROR(
        cudaMemcpyAsync(&this->network_d_->params_, &this->params_, sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));
  }
}

template <class PARAMS_T, class OUTPUT_T>
__device__ void LSTMHelper<PARAMS_T, OUTPUT_T>::initialize(float* theta_s)
{
  static_assert(std::is_trivially_copyable<PARAMS_T>::value);
  int slide = PARAMS_T::SHARED_MEM_REQUEST_GRD;
  if (PARAMS_T::SHARED_MEM_REQUEST_GRD == 0)
  {
    slide = 0;
  }
  else
  {
    PARAMS_T* shared_params = (PARAMS_T*)theta_s;
    *shared_params = this->params_;
  }
  output_nn_->initialize(theta_s + slide);

  int block_idx = (blockDim.x * threadIdx.z + threadIdx.x) * (SHARED_MEM_REQUEST_BLK) + SHARED_MEM_REQUEST_GRD;
  float* c = &theta_s[block_idx];
  float* h = &theta_s[block_idx + HIDDEN_DIM];

  if (PARAMS_T::SHARED_MEM_REQUEST_GRD == 0)
  {
    for (int i = threadIdx.y; i < HIDDEN_DIM; i += blockDim.y)
    {
      c[i] = params_.initial_cell[i];
      h[i] = params_.initial_hidden[i];
    }
  }
  else
  {
    PARAMS_T* shared_params = (PARAMS_T*)theta_s;
    for (int i = threadIdx.y; i < HIDDEN_DIM; i += blockDim.y)
    {
      c[i] = shared_params->initial_cell[i];
      h[i] = shared_params->initial_hidden[i];
    }
  }
}

template <class PARAMS_T, class OUTPUT_T>
void LSTMHelper<PARAMS_T, OUTPUT_T>::updateLSTMInitialStates(const Eigen::Ref<const hidden_state> hidden,
                                                             const Eigen::Ref<const hidden_state> cell)
{
  // TODO copy the LSTM values correctly
}

template <class PARAMS_T, class OUTPUT_T>
void LSTMHelper<PARAMS_T, OUTPUT_T>::updateOutputModel(const std::vector<int>& description,
                                                       const std::vector<float>& data)
{
  output_nn_->updateModel(description, data);
  output_nn_->paramsToDevice();
}

template <class PARAMS_T, class OUTPUT_T>
void LSTMHelper<PARAMS_T, OUTPUT_T>::updateLSTM(PARAMS_T& params)
{
  params_ = params;
  resetInitialStateCPU();
  paramsToDevice();
}

template <class PARAMS_T, class OUTPUT_T>
void LSTMHelper<PARAMS_T, OUTPUT_T>::forward(const Eigen::Ref<const input_array>& input,
                                             Eigen::Ref<output_array> output)
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

  typename OUTPUT_T::input_array nn_input;
  nn_input.head(HIDDEN_DIM) = h_next;
  nn_input.tail(INPUT_DIM) = input;

  output_nn_->forward(nn_input, output);
}

template <class PARAMS_T, class OUTPUT_T>
__device__ float* LSTMHelper<PARAMS_T, OUTPUT_T>::forward(float* input, float* theta_s)
{
  // Weights
  float* W_ii = (float*)&(this->params_.W_ii);
  float* W_im = (float*)&(this->params_.W_im);
  float* W_fi = (float*)&(this->params_.W_fi);
  float* W_fm = (float*)&(this->params_.W_fm);
  float* W_oi = (float*)&(this->params_.W_oi);
  float* W_om = (float*)&(this->params_.W_om);
  float* W_ci = (float*)&(this->params_.W_ci);
  float* W_cm = (float*)&(this->params_.W_cm);

  // Biases
  float* b_i = (float*)&(this->params_.b_i);  // hidden_size
  float* b_f = (float*)&(this->params_.b_f);  // hidden_size
  float* b_o = (float*)&(this->params_.b_o);  // hidden_size
  float* b_c = (float*)&(this->params_.b_c);  // hidden_size

  int i, j;

  // Intermediate outputs
  int block_idx = (blockDim.x * threadIdx.z + threadIdx.x) * (SHARED_MEM_REQUEST_BLK) + SHARED_MEM_REQUEST_GRD;
  float* c = &theta_s[block_idx];
  float* h = &theta_s[block_idx + HIDDEN_DIM];
  // float* next_cell_state = &theta_s[block_idx + 2 * HIDDEN_DIM];
  // float* next_hidden_state = &theta_s[block_idx + 3 * HIDDEN_DIM];
  float* g_i = &theta_s[block_idx + 4 * HIDDEN_DIM];  // input gate
  float* g_f = &theta_s[block_idx + 5 * HIDDEN_DIM];  // forget gate
  float* g_o = &theta_s[block_idx + 6 * HIDDEN_DIM];  // output gate
  float* cell_update = &theta_s[block_idx + 7 * HIDDEN_DIM];
  float* x = &theta_s[block_idx + 8 * HIDDEN_DIM];

  int tdy = threadIdx.y;

  // load input into theta_s
  for (i = tdy; i < INPUT_DIM; i += blockDim.y)
  {
    x[i] = input[i];
  }
  __syncthreads();

  int index = 0;
  // apply each gate in parallel

  for (i = tdy; i < HIDDEN_DIM; i += blockDim.y)
  {
    g_i[i] = 0;
    g_f[i] = 0;
    g_o[i] = 0;
    cell_update[i] = 0;
    for (j = 0; j < INPUT_DIM; j++)
    {
      index = i * INPUT_DIM + j;
      g_i[i] += W_ii[index] * x[j];
      g_f[i] += W_fi[index] * x[j];
      g_o[i] += W_oi[index] * x[j];
      cell_update[i] += W_ci[index] * x[j];
    }
    for (j = 0; j < HIDDEN_DIM; j++)
    {
      index = i * HIDDEN_DIM + j;
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
    cell_update[i] = TANH(cell_update[i]);
  }
  __syncthreads();

  // copy computed hidden/cell state to theta_s
  for (i = tdy; i < HIDDEN_DIM; i += blockDim.y)
  {
    c[i] = g_i[i] * cell_update[i] + g_f[i] * c[i];
    h[i] = TANH(c[i]) * g_o[i];
  }
  __syncthreads();
  // TODO make sure to shift by entire thing
  // TODO compute the output network

  int slide = PARAMS_T::SHARED_MEM_REQUEST_GRD;
  if (PARAMS_T::SHARED_MEM_REQUEST_GRD == 0)
  {
    slide = 0;
  }
  OUTPUT_PARAMS* params = (OUTPUT_PARAMS*)(theta_s + slide);

  float* output_act = &theta_s[block_idx + 8 * HIDDEN_DIM + INPUT_DIM];
  for (i = tdy; i < HIDDEN_DIM; i += blockDim.y)
  {
    output_act[i] = h[i];
  }
  for (i = tdy; i < INPUT_DIM; i += blockDim.y)
  {
    output_act[i + HIDDEN_DIM] = input[i];
  }
  __syncthreads();

  return output_nn_->forward(nullptr, output_act, params, 0);
}
template <class PARAMS_T, class OUTPUT_T>
void LSTMHelper<PARAMS_T, OUTPUT_T>::resetHiddenState()
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    hidden_state_[i] = params_.initial_hidden[i];
  }
}

template <class PARAMS_T, class OUTPUT_T>
void LSTMHelper<PARAMS_T, OUTPUT_T>::resetCellState()
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    cell_state_[i] = params_.initial_cell[i];
  }
}

template <class PARAMS_T, class OUTPUT_T>
void LSTMHelper<PARAMS_T, OUTPUT_T>::resetInitialStateCPU()
{
  for (int i = 0; i < HIDDEN_DIM; i++)
  {
    hidden_state_[i] = params_.initial_hidden[i];
    cell_state_[i] = params_.initial_cell[i];
  }
}
