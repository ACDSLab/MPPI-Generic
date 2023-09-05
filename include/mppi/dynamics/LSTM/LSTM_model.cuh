#ifndef LSTM_DYNAMICS_CUH_
#define LSTM_DYNAMICS_CUH_

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/nn_helpers/meta_math.h>
#include <mppi/utils/file_utils.h>

#include <cnpy.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>

/**
 * For AutoRally
 * State x, y, yaw, roll, u_x, u_y, yaw_rate
 * LSTMModel<7,2,3,6,32,32,4> model(dt, u_constraint);
 * DYNAMICS_DIM = 4
 */

// Including neural net model
// #ifdef MPPI_NNET_USING_CONSTANT_MEM__
// __device__ __constant__ float NNET_PARAMS[param_counter(6,32,32,4)];
// #endif

// history of states and controls - roughly 10 (initializer network) (inside paramsToDevice()?)

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER = 11, int INIT_DIM = 200>
struct LSTMDynamicsParams : public DynamicsParams
{
  enum class StateIndex : int
  {
    POS_X = 0,
    POS_Y,
    YAW,
    ROLL,
    BODY_VEL_X,
    BODY_VEL_Y,
    YAW_RATE,
    NUM_STATES = S_DIM
  };
  enum class ControlIndex : int
  {
    STEERING = 0,
    THROTTLE,
    NUM_CONTROLS = C_DIM
  };

  enum class OutputIndex : int
  {
    POS_X = 0,
    POS_Y,
    YAW,
    ROLL,
    BODY_VEL_X,
    BODY_VEL_Y,
    YAW_RATE,
    NUM_OUTPUTS = S_DIM
  };
  static const int DYNAMICS_DIM = S_DIM - K_DIM;  ///< number of inputs from state
  // static const int NUM_LAYERS = layer_counter(layer_args...); ///< Total number of layers (including in/out layer)
  // static const int PRIME_PADDING = 1; ///< Extra padding to largest layer to avoid shared mem bank conflicts
  // static const int LARGEST_LAYER = neuron_counter(layer_args...) + PRIME_PADDING; ///< Number of neurons in the
  // largest layer(including in/out neurons) static const int NUM_PARAMS = param_counter(layer_args...); ///< Total
  // number of model parameters;
  static const int HIDDEN_DIM = H_DIM;
  static const int LSTM_NUM_WEIGHTS = (4 * DYNAMICS_DIM + 4 * HIDDEN_DIM + 4) * HIDDEN_DIM;
  static const int INITIALIZATION_WEIGHTS = 2 * (BUFFER * (DYNAMICS_DIM + C_DIM) * INIT_DIM + INIT_DIM * HIDDEN_DIM);
  static const int OUTPUT_WEIGHTS = DYNAMICS_DIM * HIDDEN_DIM + DYNAMICS_DIM + INITIALIZATION_WEIGHTS;
  static const int NUM_PARAMS = LSTM_NUM_WEIGHTS + OUTPUT_WEIGHTS;
  static const int SHARED_MEM_REQUEST_GRD_BYTES = 0;  ///< Amount of shared memory we need per BLOCK.

  float theta[1] = { 0.0 };    // DO NOT USE, FOR AUTORALLY PLANT COMPATIBILITY ONLY
  int stride_idcs[1] = { 0 };  // DO NOT USE, FOR AUTORALLY PLANT COMPATIBILITY ONLY
  /** Shared memory components
   * cell state - HIDDEN_DIM
   * hidden state - HIDDEN_DIM
   * next cell state - HIDDEN_DIM
   * next hidden state - HIDDEN_DIM
   * input gate output - HIDDEN_DIM
   * forget gate output - HIDDEN_DIM
   * output gate output - HIDDEN_DIM
   * intermediate cell update output - HIDDEN_DIM
   * starting input sequence - DYNAMICS_DIM + CONTROL_DIM
   */
  static const int SHARED_MEM_REQUEST_BLK_BYTES =
      8 * HIDDEN_DIM + DYNAMICS_DIM + C_DIM * sizeof(float);  ///< Amount of shared memory we need per ROLLOUT.

  static const int HIDDEN_HIDDEN_SIZE = HIDDEN_DIM * HIDDEN_DIM;
  static const int STATE_HIDDEN_SIZE = HIDDEN_DIM * (DYNAMICS_DIM + C_DIM);
  static const int BUFFER_INTER_SIZE = BUFFER * (DYNAMICS_DIM + C_DIM) * INIT_DIM;
  static const int INTER_HIDDEN_SIZE = HIDDEN_DIM * INIT_DIM;
  typedef float HIDDEN_HIDDEN_MAT[HIDDEN_HIDDEN_SIZE];
  typedef float STATE_HIDDEN_MAT[STATE_HIDDEN_SIZE];
  // typedef int NetStructureArr[NUM_LAYERS];
  // typedef int StrideIcsArr[(NUM_LAYERS - 1) * 2];
  HIDDEN_HIDDEN_MAT W_im = { 0.0 };
  HIDDEN_HIDDEN_MAT W_fm = { 1.0, 0.5 };
  HIDDEN_HIDDEN_MAT W_om = { 0.0 };
  HIDDEN_HIDDEN_MAT W_cm = { 0.0 };
  STATE_HIDDEN_MAT W_ii = { 0.0 };
  STATE_HIDDEN_MAT W_fi = { 0.0 };
  STATE_HIDDEN_MAT W_oi = { 0.0 };
  STATE_HIDDEN_MAT W_ci = { 0.0 };
  STATE_HIDDEN_MAT W_y = { 0.0 };

  float b_i[HIDDEN_DIM] = { 0.0 };
  float b_f[HIDDEN_DIM] = { 0.0 };
  float b_o[HIDDEN_DIM] = { 0.0 };
  float b_c[HIDDEN_DIM] = { 0.0 };
  float b_y[DYNAMICS_DIM] = { 0.0 };

  float initial_hidden[HIDDEN_DIM] = { 0.0 };
  float initial_cell[HIDDEN_DIM] = { 0.0 };
  // Initialization Network - might want to make these float* and allocate size in constructor
  std::shared_ptr<float> W_hidden_input;
  std::shared_ptr<float> b_hidden_input;
  std::shared_ptr<float> W_cell_input;
  std::shared_ptr<float> b_cell_input;
  std::shared_ptr<float> W_hidden_output;
  std::shared_ptr<float> b_hidden_output;
  std::shared_ptr<float> W_cell_output;
  std::shared_ptr<float> b_cell_output;

  float buffer[(C_DIM + DYNAMICS_DIM) * BUFFER] = { 0.0 };

  // Boolean flags
  bool copy_everything = true;

  // packed by all weights that connect layer 1 to layer 2 neuron 1, bias for all connections from layer 1 to layer 2
  // then layer 2 neuron 2, etc
  // ThetaArr theta = {0.0};
  // TODO stride_idcs and net_strucutre should be write protected, so user cannot modify these values

  // index into theta for weights and bias (layer 0 weights start, no bias in input layer, layer 1 weights start, layer1
  // bias start... StrideIcsArr stride_idcs = {0};

  //[neurons in layer 1, neurons in layer 2, ...]
  // NetStructureArr net_structure = {layer_args...};

  LSTMDynamicsParams()
  {
    std::shared_ptr<float> W_hidden_input_tmp(new float[BUFFER_INTER_SIZE], std::default_delete<float[]>());
    std::shared_ptr<float> W_cell_input_tmp(new float[BUFFER_INTER_SIZE], std::default_delete<float[]>());
    std::shared_ptr<float> b_cell_input_tmp(new float[INIT_DIM], std::default_delete<float[]>());
    std::shared_ptr<float> b_hidden_input_tmp(new float[INIT_DIM], std::default_delete<float[]>());
    std::shared_ptr<float> W_hidden_output_tmp(new float[INTER_HIDDEN_SIZE], std::default_delete<float[]>());
    std::shared_ptr<float> W_cell_output_tmp(new float[INTER_HIDDEN_SIZE], std::default_delete<float[]>());

    std::shared_ptr<float> b_hidden_output_tmp(new float[HIDDEN_DIM], std::default_delete<float[]>());
    std::shared_ptr<float> b_cell_output_tmp(new float[HIDDEN_DIM], std::default_delete<float[]>());
    W_hidden_input = W_hidden_input_tmp;
    b_hidden_input = b_hidden_input_tmp;
    W_cell_input = W_cell_input_tmp;
    b_cell_input = b_cell_input_tmp;

    W_hidden_output = W_hidden_output_tmp;
    b_hidden_output = b_hidden_output_tmp;
    W_cell_output = W_cell_output_tmp;
    b_cell_output = b_cell_output_tmp;
    //   int stride = 0;
    //   for(int i = 0; i < NUM_LAYERS - 1; i++) {
    //     stride_idcs[2 * i] = stride;
    //     stride += net_structure[i+1] * net_structure[i];
    //     stride_idcs[2*i + 1] = stride;
    //     stride += net_structure[i+1];
    //   }

    for (int i = 0; i < BUFFER_INTER_SIZE; i++)
    {
      W_hidden_input.get()[i] = 0;
      W_cell_input.get()[i] = 0;
    }
    for (int i = 0; i < INTER_HIDDEN_SIZE; i++)
    {
      W_hidden_output.get()[i] = 0;
      W_cell_output.get()[i] = 0;
    }
    for (int i = 0; i < INIT_DIM; i++)
    {
      b_hidden_input.get()[i] = 0;
      b_cell_input.get()[i] = 0;
    }
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      b_hidden_output.get()[i] = 0;
      b_cell_output.get()[i] = 0;
    }
    for (int i = 0; i < STATE_HIDDEN_SIZE; i++)
    {
      W_y[i] = 0;
    }
    for (int i = 0; i < DYNAMICS_DIM; i++)
    {
      b_y[i] = 0;
    }
    for (int i = 0; i < STATE_HIDDEN_SIZE; i++)
    {
      W_ii[i] = 0;
      W_fi[i] = 0;
      W_ci[i] = 0;
      W_oi[i] = 0;
    }
    for (int i = 0; i < HIDDEN_HIDDEN_SIZE; i++)
    {
      W_im[i] = 0;
      W_fm[i] = 0;
      W_cm[i] = 0;
      W_om[i] = 0;
    }
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      b_i[i] = 0;
      b_f[i] = 0;
      b_c[i] = 0;
      b_o[i] = 0;
    }
  };

  ~LSTMDynamicsParams()
  {
    // delete W_hidden_input;
  }

  void updateBuffer(const Eigen::Matrix<float, S_DIM + C_DIM, BUFFER>& new_buffer)
  {
    int s_c_dim = DYNAMICS_DIM + C_DIM;
    for (int col = 0; col < BUFFER; col++)
    {
      for (int row = 0; row < s_c_dim; row++)
      {
        buffer[col * (s_c_dim) + row] = new_buffer(K_DIM + row, col);
      }
    }
  }

  // Calculate new initial cell and hidden state
  __host__ void updateInitialLSTMState()
  {
    for (int i = 0; i < (C_DIM + DYNAMICS_DIM) * BUFFER; i++)
    {
      if (!isfinite(buffer[i]))
      {
        std::cout << "biffer not finite " << i << " " << buffer[i] << std::endl;
      }
    }

    // Create Eigen types
    using W_input = Eigen::Matrix<float, INIT_DIM, BUFFER*(DYNAMICS_DIM + C_DIM), Eigen::RowMajor>;
    using W_output = Eigen::Matrix<float, HIDDEN_DIM, INIT_DIM, Eigen::RowMajor>;
    using input_layer = Eigen::Matrix<float, BUFFER*(DYNAMICS_DIM + C_DIM), 1>;
    using intermediate_layer = Eigen::Matrix<float, INIT_DIM, 1>;
    using output_layer = Eigen::Matrix<float, HIDDEN_DIM, 1>;

    // Input/Outputs
    Eigen::Map<const input_layer> input_mat(buffer);
    Eigen::Map<output_layer> hidden_output(initial_hidden);
    Eigen::Map<output_layer> cell_output(initial_cell);
    // Weights
    Eigen::Map<const W_input> W_hidden_input_mat(W_hidden_input.get());
    Eigen::Map<const W_input> W_cell_input_mat(W_cell_input.get());
    Eigen::Map<const W_output> W_hidden_output_mat(W_hidden_output.get());
    Eigen::Map<const W_output> W_cell_output_mat(W_cell_output.get());
    // Biases
    Eigen::Map<const intermediate_layer> b_hidden_input_mat(b_hidden_input.get());
    Eigen::Map<const intermediate_layer> b_cell_input_mat(b_cell_input.get());
    Eigen::Map<const output_layer> b_hidden_output_mat(b_hidden_output.get());
    Eigen::Map<const output_layer> b_cell_output_mat(b_cell_output.get());
    // Temporary mats
    intermediate_layer intermediate_hidden;
    intermediate_layer intermediate_cell;

    // calculate layer 1
    intermediate_hidden = W_hidden_input_mat * input_mat + b_hidden_input_mat;
    intermediate_cell = W_cell_input_mat * input_mat + b_cell_input_mat;
    if (!intermediate_hidden.allFinite())
    {
      std::cout << "W_hidden_input mat\n" << W_hidden_input_mat << std::endl;
      std::cout << "input mat\n" << input_mat << std::endl;
      std::cout << "bias input mat\n" << b_cell_input_mat << std::endl;
      std::cout << "result:\n" << intermediate_hidden << std::endl;
      std::cout << "intermediate hidden is not finite" << std::endl;
    }
    if (!intermediate_cell.allFinite())
    {
      std::cout << intermediate_cell << std::endl;
      std::cout << "intermediate cell is not finite" << std::endl;
    }
    assert(intermediate_hidden.allFinite());
    assert(intermediate_cell.allFinite());
    // relu
    // Apply ReLU with lambda (should work as this is the Eigen example)
    intermediate_hidden = intermediate_hidden.unaryExpr([](float x) { return fmaxf(0, x); });
    intermediate_cell = intermediate_cell.unaryExpr([](float x) { return fmaxf(0, x); });
    // calculate layer 2
    // std::cout << "W_hidden_input_map: " << W_hidden_input_mat.transpose() << std::endl;
    // std::cout << "intermedaite hidden: " << intermediate_hidden.transpose() << std::endl;
    // std::cout << "bias output: " << b_hidden_output_mat.transpose() << std::endl;
    hidden_output = W_hidden_output_mat * intermediate_hidden + b_hidden_output_mat;
    cell_output = W_cell_output_mat * intermediate_cell + b_cell_output_mat;
    // std::cout << "hidden: \n" << hidden_output.transpose() << std::endl;
    // std::cout << "cell: \n" << cell_output.transpose() << std::endl;
    // for(int i = 0; i < 15; i++) {
    //  std::cout << "at index " << i << ": " << initial_hidden[i] << ", " << initial_cell[i] << std::endl;
    //}
  };
};

/**
 * @file lstm_model.cuh
 * @author Bogdan Vlahov <bvlahov3@gatech.edu>
 * @date June 14, 2021
 * @copyright 2021 Georgia Institute of Technology
 * @brief Neural Network Model class definition
 * @tparam S_DIM state dimension
 * @tparam C_DIM control dimension
 * @tparam K_DIM dimensions that are ignored from the state, 1 ignores the first, 2 the first and second, etc.
 * For example in AutoRally we want to input the last 4 values of our state (dim 7), so our K is 3
 * If you use the entire state in the NN K should equal 0
 * @tparam layer_args size of the NN layers
 */

using namespace MPPI_internal;

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER = 11, int INIT_DIM = 200>
class LSTMModel : public Dynamics<LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>,
                                  LSTMDynamicsParams<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>>
{
public:
  // TODO remove duplication of calculation of values, pull from the structure
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using LSTM_MODEL = LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>;
  using LSTM_PARAMS = LSTMDynamicsParams<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>;
  using PARENT_CLASS = Dynamics<LSTM_MODEL, LSTM_PARAMS>;

  // Define Eigen fixed size matrices
  using state_array = typename PARENT_CLASS::state_array;
  using control_array = typename PARENT_CLASS::control_array;
  using output_array = typename PARENT_CLASS::output_array;
  using dfdx = typename PARENT_CLASS::dfdx;
  using dfdu = typename PARENT_CLASS::dfdu;

  static const int DYNAMICS_DIM = S_DIM - K_DIM;  ///< number of inputs from state
  // static const int NUM_LAYERS = layer_counter(layer_args...); ///< Total number of layers (including in/out layer)
  // static const int PRIME_PADDING = 1; ///< Extra padding to largest layer to avoid shared mem bank conflicts
  // static const int LARGEST_LAYER = neuron_counter(layer_args...) + PRIME_PADDING; ///< Number of neurons in the
  // largest layer(including in/out neurons) static const int NUM_PARAMS = param_counter(layer_args...); ///< Total
  // number of model parameters; static const int SHARED_MEM_REQUEST_GRD_BYTES = 0; ///< Amount of shared memory we need per
  // BLOCK.
  static const int SHARED_MEM_REQUEST_BLK_BYTES =
      8 * H_DIM + DYNAMICS_DIM + C_DIM * sizeof(float);  ///< Amount of shared memory we need per ROLLOUT.

  LSTMModel(cudaStream_t stream = 0);
  LSTMModel(std::array<float2, C_DIM> control_rngs, cudaStream_t stream = 0);

  ~LSTMModel();

  // std::array<int, NUM_LAYERS> getNetStructure() {
  //   std::array<int, NUM_LAYERS> array;
  //   for(int i = 0; i < NUM_LAYERS; i++) {
  //     array[i] = this->params_.net_structure[i];
  //   }
  //   return array;
  // }
  // std::array<int, (NUM_LAYERS - 1) * 2> getStideIdcs() {
  //   std::array<int, (NUM_LAYERS - 1) * 2> array;
  //   for(int i = 0; i < (NUM_LAYERS - 1)*2; i++) {
  //     array[i] = this->params_.stride_idcs[i];
  //   }
  //   return array;
  // }
  // std::array<float, NUM_PARAMS> getTheta() {
  //   std::array<float, NUM_PARAMS> array;
  //   for(int i = 0; i < NUM_PARAMS; i++) {
  //     array[i] = this->params_.theta[i];
  //   }
  //   return array;
  // }
  // __device__ int* getNetStructurePtr(){return this->params_.net_structure;}
  // __device__ int* getStrideIdcsPtr(){return this->params_.stride_idcs;}
  // __device__ float* getThetaPtr(){return this->params_.theta;}
  std::string getDynamicsModelName() const override
  {
    return "LSTM Autorally Model";
  }

  void CPUSetup();

  void GPUSetup()
  {
    if (!this->GPUMemStatus_)
    {
      this->model_d_ = Managed::GPUSetup(this);
    }
    else
    {
      std::cout << "GPU Memory already set" << std::endl;  // TODO should this be an exception?
    }
  }

  void paramsToDevice();

  void freeCudaMem();

  void printState();

  void printParams();

  void loadParams(const std::string& lstm_model_path, const std::string& hidden_model_path,
                  const std::string& cell_model_path, const std::string& output_model_path);

  void updateModel(std::vector<int> description, std::vector<float> data);

  // bool computeGrad(const Eigen::Ref<const state_array>& state,
  //                  const Eigen::Ref<const control_array>& control,
  //                  Eigen::Ref<dfdx> A,
  //                  Eigen::Ref<dfdu> B);

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);
  void computeKinematics(const Eigen::Ref<const state_array>& state, Eigen::Ref<state_array> s_der);

  void initializeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                          Eigen::Ref<output_array> output, float t_0, float dt);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s = nullptr);
  __device__ void computeKinematics(float* state, float* state_der);
  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt);

protected:
  Eigen::Matrix<float, H_DIM, 1> hidden_state_;
  Eigen::Matrix<float, H_DIM, 1> cell_state_;
  bool init_ = false;

private:
  // Eigen::MatrixXf* weighted_in_ = nullptr;
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* weights_ = nullptr;
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* biases_ = nullptr;
};

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
const int LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::DYNAMICS_DIM;

// template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
// const int LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::NUM_LAYERS;

// template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
// const int LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::LARGEST_LAYER;

// template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
// const int LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::NUM_PARAMS;

// template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
// const int LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::SHARED_MEM_REQUEST_GRD_BYTES;

template <int S_DIM, int C_DIM, int K_DIM, int H_DIM, int BUFFER, int INIT_DIM>
const int LSTMModel<S_DIM, C_DIM, K_DIM, H_DIM, BUFFER, INIT_DIM>::SHARED_MEM_REQUEST_BLK_BYTES;

#if __CUDACC__
#include "LSTM_model.cu"
#endif

#endif  // LSTM_DYNAMICS_CUH_
