#include "lstm_lstm_helper.cuh"

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(cudaStream_t stream)
  : Managed(stream)
{
  init_model_ = std::make_shared<INIT_T>(stream);
  lstm_ = new LSTM_T(stream);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(std::string init_path,
                                                                                         std::string lstm_path,
                                                                                         cudaStream_t stream)
  : Managed(stream)
{
  init_model_ = std::make_shared<INIT_T>(stream);
  lstm_ = new LSTM_T(stream);
  loadParamsInit(init_path);
  loadParamsLSTM(lstm_path);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::initializeLSTM(const Eigen::Ref<const init_buffer>& buffer)
{
  // reset hidden/cell state
  init_model_->resetHiddenCPU();

  int t = 0;
  for (t = 0; t < INITIAL_LEN - 1; t++)
  {
    init_model_->forward(buffer.col(t));
  }

  // run full model with output at end
  typename INIT_T::output_array output;
  init_model_->forward(buffer.col(t), output);

  // set the lstm initial hidden/cell to output
  lstm_->setHiddenState(output.head(HIDDEN_DIM));
  lstm_->setCellState(output.tail(HIDDEN_DIM));
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::loadParamsInit(const std::string& model_path)
{
  init_model_->loadParams(model_path);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::loadParamsInit(const cnpy::npz_t& npz)
{
  init_model_->loadParams(npz);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::loadParamsLSTM(const std::string& model_path)
{
  lstm_->loadParams(model_path);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::loadParamsLSTM(const cnpy::npz_t& npz)
{
  lstm_->loadParams(npz);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::GPUSetup()
{
  lstm_->GPUSetup();
  if (!this->GPUMemStatus_)
  {
    network_d_ = Managed::GPUSetup<LSTMLSTMHelper<INIT_T, LSTM_T, INIT_LEN>>(this);
    HANDLE_ERROR(cudaMemcpyAsync(&(this->network_d_->lstm_), &(this->lstm_->network_d_), sizeof(LSTM_T*),
                                 cudaMemcpyHostToDevice, this->stream_));
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;
  }
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::freeCudaMem()
{
  lstm_->freeCudaMem();
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::paramsToDevice()
{
  lstm_->paramsToDevice();
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::forward(const Eigen::Ref<const input_array>& input,
                                                          Eigen::Ref<output_array> output)
{
  lstm_->forward(input, output);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
__device__ void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::initialize(float* theta_s)
{
  lstm_->initialize(theta_s);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
__device__ float* LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::forward(float* input, float* theta_s)
{
  return lstm_->forward(input, theta_s);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::updateOutputModel(const std::vector<int>& description,
                                                                    const std::vector<float>& data)
{
  lstm_->updateOutputModel(description, data);
}
template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::updateOutputModelInit(const std::vector<int>& description,
                                                                        const std::vector<float>& data)
{
  init_model_->updateOutputModel(description, data);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
std::shared_ptr<INIT_T> LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::getInitModel()
{
  return init_model_;
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTM_T* LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::getLSTMModel()
{
  return lstm_;
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
typename INIT_T::LSTM_PARAMS_T LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::getInitParams()
{
  return init_model_->getLSTMParams();
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::setInitParams(INIT_PARAMS_T& params)
{
  init_model_->setLSTMParams(params);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
typename LSTM_T::LSTM_PARAMS_T LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::getLSTMParams()
{
  return lstm_->getLSTMParams();
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::setLSTMParams(LSTM_PARAMS_T& params)
{
  lstm_->setLSTMParams(params);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
typename LSTM_T::OUTPUT_FNN_T* LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::getOutputModel()
{
  return lstm_->getOutputModel();
}
