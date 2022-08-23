#ifndef MPPIGENERIC_LSTM_HELPER_CUH
#define MPPIGENERIC_LSTM_HELPER_CUH

#include "fnn_helper.cuh"

// TODO need copies of past hidden/cell and initial hidden/cell
template<int INPUT_SIZE, int H_SIZE, int USE_SHARED=1>
struct LSTMParams {
  static const int HIDDEN_DIM = H_SIZE;
  static const int INPUT_DIM = INPUT_SIZE;

  static const int SHARED_MEM_REQUEST_BLK =
          8 * HIDDEN_DIM + INPUT_DIM;  ///< Amount of shared memory we need per ROLLOUT.
  static const int SHARED_MEM_REQUEST_GRD = sizeof(LSTMParams<INPUT_SIZE, H_SIZE>) * USE_SHARED;  ///< Amount of shared memory we need per BLOCK.

  static const int HIDDEN_HIDDEN_SIZE = HIDDEN_DIM * HIDDEN_DIM;
  static const int INPUT_HIDDEN_SIZE = HIDDEN_DIM * (INPUT_DIM);
  typedef float HIDDEN_HIDDEN_MAT[HIDDEN_HIDDEN_SIZE];
  typedef float INPUT_HIDDEN_MAT[INPUT_HIDDEN_SIZE];

  HIDDEN_HIDDEN_MAT W_im = { 0.0f };
  HIDDEN_HIDDEN_MAT W_fm = { 0.0f };
  HIDDEN_HIDDEN_MAT W_om = { 0.0f };
  HIDDEN_HIDDEN_MAT W_cm = { 0.0f };
  INPUT_HIDDEN_MAT W_ii = { 0.0f };
  INPUT_HIDDEN_MAT W_fi = { 0.0f };
  INPUT_HIDDEN_MAT W_oi = { 0.0f };
  INPUT_HIDDEN_MAT W_ci = { 0.0f };

  float b_i[HIDDEN_DIM] = { 0.0f };
  float b_f[HIDDEN_DIM] = { 0.0f };
  float b_o[HIDDEN_DIM] = { 0.0f };
  float b_c[HIDDEN_DIM] = { 0.0f };

  float initial_hidden[HIDDEN_DIM] = { 0.0f };
  float initial_cell[HIDDEN_DIM] = { 0.0f };

  void setAllValues(float input) {
    for(int i = 0; i < HIDDEN_HIDDEN_SIZE; i++) {
      W_im[i] = input;
      W_fm[i] = input;
      W_om[i] = input;
      W_cm[i] = input;
    }
    for(int i = 0; i < INPUT_HIDDEN_SIZE; i++) {
      W_ii[i] = input;
      W_fi[i] = input;
      W_oi[i] = input;
      W_ci[i] = input;
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
      b_i[i] = input;
      b_f[i] = input;
      b_o[i] = input;
      b_c[i] = input;
      initial_hidden[i] = input;
      initial_cell[i] = input;
    }
  }
};

template<class PARAMS_T, class OUTPUT_T>
class LSTMHelper : public Managed
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static const int NUM_PARAMS = PARAMS_T::NUM_PARAMS;   ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD = PARAMS_T::SHARED_MEM_REQUEST_GRD + OUTPUT_T::SHARED_MEM_REQUEST_GRD; ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK = PARAMS_T::SHARED_MEM_REQUEST_BLK + OUTPUT_T::SHARED_MEM_REQUEST_BLK;  ///< Amount of shared memory we need per ROLLOUT.

  static const int INPUT_DIM = PARAMS_T::INPUT_DIM;
  static const int HIDDEN_DIM = PARAMS_T::HIDDEN_DIM;
  static const int OUTPUT_DIM = OUTPUT_T::OUTPUT_DIM;

  typedef Eigen::Matrix<float, PARAMS_T::INPUT_DIM, 1> input_array;
  typedef Eigen::Matrix<float, OUTPUT_T::OUTPUT_DIM, 1> output_array;
  typedef Eigen::Matrix<float, OUTPUT_T::OUTPUT_DIM, PARAMS_T::INPUT_DIM> dfdx;
  typedef PARAMS_T LSTM_PARAMS_T;
  typedef OUTPUT_T OUTPUT_FNN_T;
  typedef typename OUTPUT_T::NN_PARAMS_T OUTPUT_PARAMS;

  using W_hh = Eigen::Matrix<float, HIDDEN_DIM, HIDDEN_DIM, Eigen::RowMajor>;
  using W_hi = Eigen::Matrix<float, HIDDEN_DIM, INPUT_DIM, Eigen::RowMajor>;
  using hidden_state = Eigen::Matrix<float, HIDDEN_DIM, 1>;

  LSTMHelper<PARAMS_T, OUTPUT_FNN_T>(cudaStream_t stream=0);
  LSTMHelper<PARAMS_T, OUTPUT_FNN_T>(std::string, cudaStream_t stream=0);

  void loadParams(const std::string& model_path);
  void loadParams(const cnpy::npz_t& npz);

  __device__ __host__ PARAMS_T getLSTMParams() {
    return params_;
  }

  __device__ void initialize(float* theta_s);

  void GPUSetup();
  void freeCudaMem();
  void paramsToDevice();

  void updateOutputModel(const std::vector<int>& description, const std::vector<float>& data);
  void updateLSTM(PARAMS_T& params);
  void updateLSTMInitialStates(const Eigen::Ref<const hidden_state> hidden, const Eigen::Ref<const hidden_state> cell);

  bool computeGrad(Eigen::Ref<dfdx> A);
  bool computeGrad(const Eigen::Ref<const input_array>& input,
                   Eigen::Ref<dfdx> A);

  void forward(const Eigen::Ref<const input_array>& input, Eigen::Ref<output_array> output);
  void forward(const Eigen::Ref<const input_array>& input);
  __device__ float* forward(float* input, float* theta_s);

  void resetHiddenCPU();
  void resetCellCPU();
  void resetHiddenCellCPU();

  hidden_state getHiddenState() {
    return hidden_state_;
  }
  hidden_state getCellState() {
    return cell_state_;
  }

  void setHiddenState(const Eigen::Ref<const hidden_state> hidden_state);
  void setCellState(const Eigen::Ref<const hidden_state> hidden_state);

  // device pointer, null on the device
  OUTPUT_FNN_T* output_nn_ = nullptr;
  LSTMHelper<PARAMS_T, OUTPUT_FNN_T>* network_d_ = nullptr;
private:
  // params
  PARAMS_T params_;

  hidden_state hidden_state_;
  hidden_state cell_state_;
};

#if __CUDACC__
#include "lstm_helper.cu"
#endif

#endif  // MPPIGENERIC_LSTM_HELPER_CUH
