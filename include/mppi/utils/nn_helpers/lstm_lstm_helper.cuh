//
// Created by jason on 8/23/22.
//

#ifndef MPPIGENERIC_LSTM_LSTM_HELPER_CUH
#define MPPIGENERIC_LSTM_LSTM_HELPER_CUH

#include "lstm_helper.cuh"

template<class INIT_T, class LSTM_T, int INITIAL_LEN>
class LSTMLSTMHelper : public Managed
{
public:
  static_assert(true);
  static const int NUM_PARAMS = INIT_T::NUM_PARAMS + LSTM_T::NUM_PARAMS;   ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD = LSTM_T::SHARED_MEM_REQUEST_GRD; ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK = LSTM_T::SHARED_MEM_REQUEST_BLK;  ///< Amount of shared memory we need per ROLLOUT.

  static const int INPUT_DIM = LSTM_T::INPUT_DIM;
  static const int HIDDEN_DIM = LSTM_T::HIDDEN_DIM;
  static const int OUTPUT_DIM = LSTM_T::OUTPUT_DIM;
  static const int INIT_LEN = INITIAL_LEN;

  static const int INIT_INPUT_DIM = INIT_T::INPUT_DIM;
  static const int INIT_HIDDEN_DIM = INIT_T::HIDDEN_DIM;

  typedef Eigen::Matrix<float, INIT_T::INPUT_DIM, INIT_LEN> init_buffer;
  typedef Eigen::Matrix<float, LSTM_T::INPUT_DIM, 1> input_array;
  typedef Eigen::Matrix<float, LSTM_T::OUTPUT_DIM, 1> output_array;
  //typedef Eigen::Matrix<float, LSTM_T::OUTPUT_DIM, PARAMS_T::INPUT_DIM> dfdx;

  // using W_hh = Eigen::Matrix<float, HIDDEN_DIM, HIDDEN_DIM, Eigen::RowMajor>;
  // using W_hi = Eigen::Matrix<float, HIDDEN_DIM, INPUT_DIM, Eigen::RowMajor>;
  using hidden_state = Eigen::Matrix<float, HIDDEN_DIM, 1>;

  LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(cudaStream_t = 0);
  LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(std::string, cudaStream_t = 0);

  void loadParams(const std::string& model_path);
  void loadParams(const cnpy::npz_t& npz);

  __device__ void initialize(float* theta_s);

  void GPUSetup();
  void freeCudaMem();
  void paramsToDevice();

  void initializeLSTM(const Eigen::Ref<const init_buffer>& buffer);

  void forward(const Eigen::Ref<const input_array>& input, Eigen::Ref<output_array> output);
  __device__ float* forward(float* input, float* theta_s);

  std::shared_ptr<INIT_T> getInitModel() {
    return init_model_;
  }
  LSTM_T* getLSTMModel() {
    return lstm_;
  }


  typename INIT_T::LSTM_PARAMS_T getInitParams() {
    return init_model_->getLSTMParams();
  }
  void setInitParams(typename INIT_T::LSTM_PARAMS_T& params) {
    init_model_->updateLSTM(params);
  }

  LSTMLSTMHelper<INIT_T, LSTM_T, INIT_LEN>* network_d_ = nullptr;
  LSTM_T* lstm_ = nullptr;
private:
  std::shared_ptr<INIT_T> init_model_ = nullptr;
};

#if __CUDACC__
#include "lstm_lstm_helper.cu"
#endif

#endif  // MPPIGENERIC_LSTM_LSTM_HELPER_CUH
