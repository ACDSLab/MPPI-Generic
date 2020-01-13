#ifndef AR_NN_DYNAMICS_CUH_
#define AR_NN_DYNAMICS_CUH_

#include <dynamics/dynamics.cuh>
#include "meta_math.h"

/**
 * For AutoRally
 * State x, y, yaw, roll, u_x, u_y, yaw_rate
 * NeuralNetModel<7,2,3,6,32,32,4> model(dt, u_constraint);
 * DYNAMICS_DIM = 4
 */


/**
 * @file neural_net_model.cuh
 * @author Grady Williams <gradyrw@gmail.com>
 * @date May 24, 2017
 * @copyright 2017 Georgia Institute of Technology
 * @brief Neural Network Model class definition
 * @tparam S_DIM state dimension
 * @tparam C_DIM control dimension
 * @tparam K_DIM dimensions that are actually used for input to NN
 * For example in AutoRally we want to input the last 4 values of our state (dim 7), so our K is 3
 * If you use the entire state in the NN K should equal 0
 * @tparam layer_args size of the NN layers
 */
template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
class NeuralNetModel : public Dynamics<S_DIM, C_DIM> {
public:
  //static const int STATE_DIM = S_DIM;
  //static const int CONTROL_DIM = C_DIM;
  static const int DYNAMICS_DIM = S_DIM - K_DIM;
  static const int NUM_LAYERS = layer_counter(layer_args...); ///< Total number of layers (including in/out layer)
  static const int PRIME_PADDING = 1; ///< Extra padding to largest layer to avoid shared mem bank conflicts
  static const int LARGEST_LAYER = neuron_counter(layer_args...) + PRIME_PADDING; ///< Number of neurons in the largest layer(including in/out neurons)
  static const int NUM_PARAMS = param_counter(layer_args...); ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD = 0; ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK = 2*LARGEST_LAYER; ///< Amount of shared memory we need per ROLLOUT.

  //NeuralNetModel<S_DIM, C_DIM, K_DIM, int... layer_args>* model_d_;

  //float* theta_d_; ///< GPU memory parameter array.
  //int* stride_idcs_d_; ///< GPU memory for keeping track of parameter strides
  //int* net_structure_d_; ///GPU memory for keeping track of the neural net structure.
  //float2* control_rngs_;
  std::array<float2, C_DIM> control_rngs_;

  //Eigen::Matrix<float, STATE_DIM, 1> state_der_; ///< The state derivative.
  //float2* control_rngs_d_;

  //Eigen::MatrixXf ip_delta_; ///< The neural net state derivative.
  //Eigen::Matrix<float, S_DIM, S_DIM + C_DIM> jac_; //Total state derivative

  NeuralNetModel(float delta_t);
  NeuralNetModel(float delta_t, std::array<float2, C_DIM> control_rngs);

  ~NeuralNetModel();

  std::array<float2, C_DIM> getControlRanges() {return control_rngs_;}

  /*

  void loadParams(std::string model_path);

  void setParams(Eigen::Matrix<float, -1, -1, Eigen::RowMajor>* weights,
                 Eigen::Matrix<float, -1, -1, Eigen::RowMajor>* biases);

  void paramsToDevice();

  void printParamVec();

  void freeCudaMem();

  void enforceConstraints(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  void updateState(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  void computeKinematics(Eigen::MatrixXf &state);

  void computeDynamics(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  void updateModel(std::vector<int> description, std::vector<float> data);

  __device__ void computeKinematics(float* state, float* state_der);

  __device__ void cudaInit(float* theta_s);

  __device__ void enforceConstraints(float* state, float* control);

  __device__ void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s);

  __device__ void incrementState(float* state, float* state_der);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s);

  __device__ void printCudaParamVec();
   */

private:
  float dt_;

  //Neural net structure
  //int net_structure_[NUM_LAYERS] = {layer_args...};
  //int stride_idcs_[NUM_LAYERS*2 + 1] = {0};

  //Host fields
  //Eigen::Matrix<float, -1, -1, Eigen::RowMajor>* weights_; ///< Matrices of weights {W_1, W_2, ... W_n}
  //Eigen::Matrix<float, -1, -1, Eigen::RowMajor>* biases_; ///< Vectors of biases {b_1, b_2, ... b_n}

  //Eigen::MatrixXf* weighted_in_;

  //float* net_params_;
};

/*

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::STATE_DIM;

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::CONTROL_DIM;

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::DYNAMICS_DIM;

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NUM_LAYERS;

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::LARGEST_LAYER;

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NUM_PARAMS;

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::SHARED_MEM_REQUEST_GRD;

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::SHARED_MEM_REQUEST_BLK;

 */

#if __CUDACC__
#include "ar_nn_model.cu"
#endif

#endif // AR_NN_DYNAMICS_CUH_