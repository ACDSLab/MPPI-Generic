#ifndef MPPIGENERIC_LSTM_HELPER_CUH
#define MPPIGENERIC_LSTM_HELPER_CUH

#include "fnn_helper.cuh"

template <bool USE_SHARED = true>
class LSTMHelper : public Managed
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef FNNHelper<USE_SHARED> OUTPUT_FNN_T;

  // TODO create destructor that deallocates memory

  LSTMHelper<USE_SHARED>(int input_dim, int hidden_dim, std::vector<int>& output_layers, cudaStream_t stream = 0);
  LSTMHelper<USE_SHARED>(std::string, cudaStream_t stream = 0);
  LSTMHelper<USE_SHARED>(const cnpy::npz_t& param_dict, std::string prefix, bool add_slash = true,
                         cudaStream_t stream = 0);

  void loadParams(const std::string& model_path);
  void loadParams(const cnpy::npz_t& npz);
  void loadParams(std::string prefix, const cnpy::npz_t& npz, bool add_slash = true);

  __device__ void initialize(float* theta_s);
  __device__ void initialize(float* theta_s, int blk_size, int grd_size, int offset);

  void GPUSetup();
  void freeCudaMem();
  void paramsToDevice();
  void copyHiddenCellToDevice();

  void updateOutputModel(const std::vector<int>& description, const std::vector<float>& data);
  void updateOutputModel(const std::vector<float>& data);
  void updateLSTMInitialStates(const Eigen::Ref<const Eigen::VectorXf> hidden,
                               const Eigen::Ref<const Eigen::VectorXf> cell);

  // bool computeGrad(Eigen::Ref<dfdx> A);
  // bool computeGrad(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<dfdx> A);

  void forward(const Eigen::Ref<const Eigen::VectorXf>& input, Eigen::Ref<Eigen::VectorXf> output);
  void forward(const Eigen::Ref<const Eigen::VectorXf>& input);
  __device__ float* forward(float* input, float* theta_s);
  __device__ float* forward(float* input, float* theta_s, float* block_ptr);
  __device__ float* forward(float* input, float* theta_s, float* hidden_cell, float* block_ptr);
  __device__ float* getInputLocation(float* theta_s);
  __device__ float* getInputLocation(float* theta_s, const int grd_shift, int blk_bytes, int shift);

  void resetHiddenCPU();
  void resetCellCPU();
  void resetHiddenCellCPU();

  void setAllValues(std::vector<float>& lstm, std::vector<float>& output)
  {
    assert(lstm.size() == getNumParams());
    memcpy(this->weights_, lstm.data(), LSTM_PARAM_SIZE_BYTES + HIDDEN_DIM * 2 * sizeof(float));
    copyWeightsToEigen();
    resetHiddenCellCPU();
    output_nn_->updateModel(output);
  }

  void setAllValues(float input)
  {
    for (int i = 0; i < LSTM_PARAM_SIZE_BYTES / sizeof(float) + 2 * HIDDEN_DIM; i++)
    {
      weights_[i] = input;
    }
    copyWeightsToEigen();
    resetHiddenCellCPU();
    output_nn_->setAllWeights(input);
  }

  Eigen::MatrixXf getHiddenState()
  {
    return hidden_state_;
  }
  Eigen::MatrixXf getCellState()
  {
    return cell_state_;
  }

  __host__ __device__ OUTPUT_FNN_T* getOutputModel()
  {
    return output_nn_;
  }

  __host__ __device__ float* getOutputWeights()
  {
    return output_nn_->weights_d_;
  }

  __host__ __device__ int getHiddenDim()
  {
    return HIDDEN_DIM;
  }
  __host__ __device__ int getInputDim()
  {
    return INPUT_DIM;
  }
  __host__ __device__ int getOutputDim()
  {
    return OUTPUT_DIM;
  }
  __host__ __device__ int getHiddenHiddenSize()
  {
    return HIDDEN_HIDDEN_SIZE;
  }
  __host__ __device__ int getInputHiddenSize()
  {
    return INPUT_HIDDEN_SIZE;
  }
  __host__ __device__ int getOutputGrdSharedSizeBytes()
  {
    return output_nn_->getGrdSharedSizeBytes();
  }
  __host__ __device__ int getLSTMGrdSharedSizeBytes()
  {
    return LSTM_SHARED_MEM_GRD_BYTES;
  }
  __host__ __device__ int getBlkLSTMSharedSizeBytes()
  {
    return (3 * HIDDEN_DIM + INPUT_DIM) * sizeof(float);
  }
  __host__ __device__ float* getWeights()
  {
    return weights_;
  }

  int getNumParams()
  {
    return LSTM_PARAM_SIZE_BYTES / sizeof(float) + 2 * HIDDEN_DIM;
  }

  Eigen::VectorXf getInputVector()
  {
    return Eigen::VectorXf(INPUT_DIM, 1);
  }
  Eigen::VectorXf getOutputVector()
  {
    return Eigen::VectorXf(OUTPUT_DIM, 1);
  }

  void copyWeightsToEigen();

  void setHiddenState(const Eigen::Ref<const Eigen::VectorXf> hidden_state);
  void setCellState(const Eigen::Ref<const Eigen::VectorXf> hidden_state);

  // device pointer, null on the device
  LSTMHelper<USE_SHARED>* network_d_ = nullptr;
  float* weights_d_ = nullptr;

private:
  // params
  OUTPUT_FNN_T* output_nn_ = nullptr;

  int HIDDEN_DIM = 0;
  int INPUT_DIM = 0;
  int OUTPUT_DIM = 0;

  int NUM_PARAMS = 1;

  int LSTM_PARAM_SIZE_BYTES = 0;
  int LSTM_SHARED_MEM_GRD_BYTES = 0;  ///< Amount of shared memory the LSTM needs per block

  int HIDDEN_HIDDEN_SIZE = HIDDEN_DIM * HIDDEN_DIM;
  int INPUT_HIDDEN_SIZE = HIDDEN_DIM * (INPUT_DIM);

  float* W_im_ = nullptr;  ///< HIDDEN_HIDDEN_SIZE
  float* W_fm_ = nullptr;  ///< HIDDEN_HIDDEN_SIZE
  float* W_om_ = nullptr;  ///< HIDDEN_HIDDEN_SIZE
  float* W_cm_ = nullptr;  ///< HIDDEN_HIDDEN_SIZE

  float* W_ii_ = nullptr;  ///< INPUT_HIDDEN_SIZE
  float* W_fi_ = nullptr;  ///< INPUT_HIDDEN_SIZE
  float* W_oi_ = nullptr;  ///< INPUT_HIDDEN_SIZE
  float* W_ci_ = nullptr;  ///< INPUT_HIDDEN_SIZE

  float* b_i_ = nullptr;   ///< HIDDEN_SIZE
  float* b_f_ = nullptr;   ///< HIDDEN_SIZE
  float* b_o_ = nullptr;   ///< HIDDEN_SIZE
  float* b_c_ = nullptr;   ///< HIDDEN_SIZE

  float* initial_hidden_ = nullptr;
  float* initial_cell_ = nullptr;

  float* weights_ = nullptr;

  Eigen::VectorXf hidden_state_;
  Eigen::VectorXf cell_state_;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_im_;  ///< HIDDEN_HIDDEN_SIZE
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_fm_;  ///< HIDDEN_HIDDEN_SIZE
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_om_;  ///< HIDDEN_HIDDEN_SIZE
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_cm_;  ///< HIDDEN_HIDDEN_SIZE

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_ii_;  ///< INPUT_HIDDEN_SIZE
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_fi_;  ///< INPUT_HIDDEN_SIZE
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_oi_;  ///< INPUT_HIDDEN_SIZE
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eig_W_ci_;  ///< INPUT_HIDDEN_SIZE

  Eigen::VectorXf eig_b_i_;                                                         ///< HIDDEN_SIZE
  Eigen::VectorXf eig_b_f_;                                                         ///< HIDDEN_SIZE
  Eigen::VectorXf eig_b_o_;                                                         ///< HIDDEN_SIZE
  Eigen::VectorXf eig_b_c_;                                                         ///< HIDDEN_SIZE

  void setupMemory(int input_dim, int hidden_dim);
};

#if __CUDACC__
#include "lstm_helper.cu"
#endif

#endif  // MPPIGENERIC_LSTM_HELPER_CUH
