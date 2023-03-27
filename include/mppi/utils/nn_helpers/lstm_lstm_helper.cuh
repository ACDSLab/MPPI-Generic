//
// Created by jason on 8/23/22.
//

#ifndef MPPIGENERIC_LSTM_LSTM_HELPER_CUH
#define MPPIGENERIC_LSTM_LSTM_HELPER_CUH

#include "lstm_helper.cuh"

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
class LSTMLSTMHelper
{
public:
  static const int NUM_PARAMS = INIT_T::NUM_PARAMS + LSTM_T::NUM_PARAMS;  ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD_BYTES =
      LSTM_T::SHARED_MEM_REQUEST_GRD_BYTES;  ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK_BYTES =
      LSTM_T::SHARED_MEM_REQUEST_BLK_BYTES;  ///< Amount of shared memory we need per ROLLOUT.

  static const int INPUT_DIM = LSTM_T::INPUT_DIM;
  static const int HIDDEN_DIM = LSTM_T::HIDDEN_DIM;
  static const int OUTPUT_DIM = LSTM_T::OUTPUT_DIM;
  static const int INIT_LEN = INITIAL_LEN;

  static const int INIT_INPUT_DIM = INIT_T::INPUT_DIM;
  static const int INIT_HIDDEN_DIM = INIT_T::HIDDEN_DIM;

  typedef Eigen::Matrix<float, INIT_T::INPUT_DIM, INIT_LEN> init_buffer;
  typedef Eigen::Matrix<float, LSTM_T::INPUT_DIM, 1> input_array;
  typedef Eigen::Matrix<float, LSTM_T::OUTPUT_DIM, 1> output_array;

  typedef typename LSTM_T::OUTPUT_FNN_T OUTPUT_FNN_T;
  typedef typename LSTM_T::LSTM_PARAMS_T LSTM_PARAMS_T;
  typedef typename INIT_T::LSTM_PARAMS_T INIT_PARAMS_T;
  typedef typename LSTM_T::OUTPUT_PARAMS_T OUTPUT_PARAMS_T;
  typedef typename INIT_T::OUTPUT_PARAMS_T INIT_OUTPUT_PARAMS_T;
  // typedef Eigen::Matrix<float, LSTM_T::OUTPUT_DIM, PARAMS_T::INPUT_DIM> dfdx;

  LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(cudaStream_t = 0);
  LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(std::string init_path, std::string lstm_path, cudaStream_t = 0);
  LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(std::string path, cudaStream_t = 0);

  void loadParams(std::string prefix, std::string path);

  void loadParamsInit(const std::string& model_path);
  void loadParamsInit(const cnpy::npz_t& npz);

  void loadParamsLSTM(const std::string& model_path);
  void loadParamsLSTM(const cnpy::npz_t& npz);

  void updateOutputModelInit(const std::vector<int>& description, const std::vector<float>& data);
  void updateOutputModel(const std::vector<int>& description, const std::vector<float>& data);

  void GPUSetup();
  void freeCudaMem();

  void resetInitHiddenCPU()
  {
    init_model_->resetHiddenCellCPU();
  }
  void resetLSTMHiddenCellCPU()
  {
    lstm_->resetHiddenCellCPU();
  }

  void initializeLSTM(const Eigen::Ref<const init_buffer>& buffer);

  void forward(const Eigen::Ref<const input_array>& input, Eigen::Ref<output_array> output);

  std::shared_ptr<INIT_T> getInitModel();
  std::shared_ptr<LSTM_T> getLSTMModel();
  LSTM_T* getLSTMDevicePtr()
  {
    return lstm_->network_d_;
  }

  void setInitParams(INIT_PARAMS_T& params);
  void setLSTMParams(LSTM_PARAMS_T& params);

  INIT_PARAMS_T getInitLSTMParams()
  {
    // TODO why using the getter method causes memory issues with compilation
    return init_model_->params_;
  }
  LSTM_PARAMS_T getLSTMParams()
  {
    return lstm_->params_;
  }
  OUTPUT_FNN_T* getOutputModel()
  {
    return lstm_->getOutputModel();
  }

private:
  std::shared_ptr<INIT_T> init_model_ = nullptr;
  std::shared_ptr<LSTM_T> lstm_ = nullptr;
};

#if __CUDACC__
#include "lstm_lstm_helper.cu"
#endif

#endif  // MPPIGENERIC_LSTM_LSTM_HELPER_CUH
