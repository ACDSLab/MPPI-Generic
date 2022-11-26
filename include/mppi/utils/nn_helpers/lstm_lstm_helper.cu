#include "lstm_lstm_helper.cuh"

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::LSTMLSTMHelper(cudaStream_t stream)
{
  init_model_ = std::make_shared<INIT_T>();
  lstm_ = std::make_shared<LSTM_T>(stream);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::LSTMLSTMHelper(std::string init_path, std::string lstm_path,
                                                            cudaStream_t stream)
{
  init_model_ = std::make_shared<INIT_T>(init_path);
  lstm_ = std::make_shared<LSTM_T>(lstm_path, stream);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::LSTMLSTMHelper(std::string path, cudaStream_t stream)
{
  init_model_ = std::make_shared<INIT_T>();
  lstm_ = std::make_shared<LSTM_T>(stream);
  loadParams("model", path);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::initializeLSTM(const Eigen::Ref<const init_buffer>& buffer)
{
  // reset hidden/cell state
  init_model_->resetHiddenCellCPU();

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

  // only copies the hidden/cell states
  lstm_->copyHiddenCellToDevice();
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
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::loadParams(std::string prefix, std::string model_path)
{
  if (!prefix.empty() && *prefix.rbegin() != '/')
  {
    prefix.append("/");
  }

  if (!fileExists(model_path))
  {
    std::cerr << "Could not load neural net model at path: " << model_path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(model_path);

  lstm_->loadParams(prefix, param_dict);
  init_model_->loadParams(prefix + "init_", param_dict, false);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::GPUSetup()
{
  lstm_->GPUSetup();
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::freeCudaMem()
{
  lstm_->freeCudaMem();
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::forward(const Eigen::Ref<const input_array>& input,
                                                          Eigen::Ref<output_array> output)
{
  lstm_->forward(input, output);
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
std::shared_ptr<LSTM_T> LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::getLSTMModel()
{
  return lstm_;
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::setInitParams(INIT_PARAMS_T& params)
{
  init_model_->setLSTMParams(params);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::setLSTMParams(LSTM_PARAMS_T& params)
{
  lstm_->setLSTMParams(params);
}
