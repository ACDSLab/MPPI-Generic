//
// Created by jason on 8/23/22.
//

#ifndef MPPIGENERIC_LSTM_LSTM_HELPER_CUH
#define MPPIGENERIC_LSTM_LSTM_HELPER_CUH

#include "lstm_helper.cuh"

// TODO need to know the expected len of the initilization
// TODO the input buffer is a different tau than the one this could want to use, use last N steps

template <bool USE_SHARED = true>
class LSTMLSTMHelper
{
public:
  // static const int NUM_PARAMS = 0;  ///< Total number of model parameters;

  // static const int INPUT_DIM = LSTM_T::INPUT_DIM;
  // static const int HIDDEN_DIM = LSTM_T::HIDDEN_DIM;
  // static const int OUTPUT_DIM = LSTM_T::OUTPUT_DIM;
  // static const int INIT_LEN = INITIAL_LEN;

  // static const int INIT_INPUT_DIM = INIT_T::INPUT_DIM;
  // static const int INIT_HIDDEN_DIM = INIT_T::HIDDEN_DIM;

  // typedef Eigen::Matrix<float, INIT_T::INPUT_DIM, INIT_LEN> init_buffer;
  // typedef Eigen::Matrix<float, LSTM_T::INPUT_DIM, 1> input_array;
  // typedef Eigen::Matrix<float, LSTM_T::OUTPUT_DIM, 1> output_array;

  LSTMLSTMHelper<USE_SHARED>(int init_input_dim, int init_hidden_dim, std::vector<int> init_output_layers,
                             int input_dim, int hidden_dim, std::vector<int> output_layers, int init_len,
                             cudaStream_t = 0);
  // LSTMLSTMHelper<USE_SHARED>(std::string init_path, std::string lstm_path, cudaStream_t = 0);
  LSTMLSTMHelper<USE_SHARED>(std::string path, std::string prefix, cudaStream_t = 0);
  LSTMLSTMHelper<USE_SHARED>(std::string path, cudaStream_t = 0);

  void loadParams(std::string prefix, std::string path);

  void loadParamsInit(const std::string& model_path);
  void loadParamsInit(const cnpy::npz_t& npz);

  void loadParamsLSTM(const std::string& model_path);
  void loadParamsLSTM(const cnpy::npz_t& npz);

  void updateOutputModelInit(const std::vector<int>& description, const std::vector<float>& data);
  void updateOutputModel(const std::vector<int>& description, const std::vector<float>& data);

  void updateOutputModelInit(const std::vector<float>& data);
  void updateOutputModel(const std::vector<float>& data);

  void GPUSetup();
  void freeCudaMem();

  Eigen::VectorXf getInputVector()
  {
    return lstm_->getInputVector();
  }
  Eigen::VectorXf getOutputVector()
  {
    return lstm_->getOutputVector();
  }

  void resetInitHiddenCPU()
  {
    init_model_->resetHiddenCellCPU();
  }
  void resetLSTMHiddenCellCPU()
  {
    lstm_->resetHiddenCellCPU();
  }

  Eigen::MatrixXf getBuffer()
  {
    return Eigen::MatrixXf(init_model_->getInputDim(), init_len_);
  }

  Eigen::MatrixXf getBuffer(int length)
  {
    return Eigen::MatrixXf(init_model_->getInputDim(), length);
  }

  void initializeLSTM(const Eigen::Ref<const Eigen::MatrixXf>& buffer);

  void forward(const Eigen::Ref<const Eigen::VectorXf>& input, Eigen::Ref<Eigen::VectorXf> output);

  std::shared_ptr<LSTMHelper<false>> getInitModel();
  std::shared_ptr<LSTMHelper<USE_SHARED>> getLSTMModel();
  LSTMHelper<USE_SHARED>* getLSTMDevicePtr()
  {
    return lstm_->network_d_;
  }

  int getInitLen()
  {
    return init_len_;
  }

private:
  std::shared_ptr<LSTMHelper<false>> init_model_ = nullptr;
  std::shared_ptr<LSTMHelper<USE_SHARED>> lstm_ = nullptr;
  int init_len_ = -1;
};

#if __CUDACC__
#include "lstm_lstm_helper.cu"
#endif

#endif  // MPPIGENERIC_LSTM_LSTM_HELPER_CUH
