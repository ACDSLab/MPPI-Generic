#ifndef AR_NN_DYNAMICS_CUH_
#define AR_NN_DYNAMICS_CUH_

#include <mppi/dynamics/dynamics.cuh>
#include "meta_math.h"
#include <mppi/utils/file_utils.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>
#include <cnpy.h>

/**
 * For AutoRally
 * State x, y, yaw, roll, u_x, u_y, yaw_rate
 * NeuralNetModel<7,2,3,6,32,32,4> model(dt, u_constraint);
 * DYNAMICS_DIM = 4
 */
#define MPPI_NNET_NONLINEARITY(ans) tanh(ans)
#define MPPI_NNET_NONLINEARITY_DERIV(ans) (1 - powf(tanh(ans), 2))

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
struct NNDynamicsParams {
  static const int DYNAMICS_DIM = S_DIM - K_DIM; ///< number of inputs from state
  static const int NUM_LAYERS = layer_counter(layer_args...); ///< Total number of layers (including in/out layer)
  static const int PRIME_PADDING = 1; ///< Extra padding to largest layer to avoid shared mem bank conflicts
  static const int LARGEST_LAYER = neuron_counter(layer_args...) + PRIME_PADDING; ///< Number of neurons in the largest layer(including in/out neurons)
  static const int NUM_PARAMS = param_counter(layer_args...); ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD = 0; ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK = 2*LARGEST_LAYER; ///< Amount of shared memory we need per ROLLOUT.

  typedef float ThetaArr[NUM_PARAMS];
  typedef int NetStructureArr[NUM_LAYERS];
  typedef int StrideIcsArr[(NUM_LAYERS - 1) * 2];

  // packed by all weights that connect layer 1 to layer 2 neuron 1, bias for all connections from layer 1 to layer 2
  // then layer 2 neuron 2, etc
  ThetaArr theta = {0.0};
  // TODO stride_idcs and net_strucutre should be write protected, so user cannot modify these values

  // index into theta for weights and bias (layer 0 weights start, no bias in input layer, layer 1 weights start, layer1 bias start...
  StrideIcsArr stride_idcs = {0};

  //[neurons in layer 1, neurons in layer 2, ...]
  NetStructureArr net_structure = {layer_args...};

  NNDynamicsParams() {
    int stride = 0;
    for(int i = 0; i < NUM_LAYERS - 1; i++) {
      stride_idcs[2 * i] = stride;
      stride += net_structure[i+1] * net_structure[i];
      stride_idcs[2*i + 1] = stride;
      stride += net_structure[i+1];
    }
  }

};

/**
 * @file neural_net_model.cuh
 * @author Grady Williams <gradyrw@gmail.com>
 * @date May 24, 2017
 * @copyright 2017 Georgia Institute of Technology
 * @brief Neural Network Model class definition
 * @tparam S_DIM state dimension
 * @tparam C_DIM control dimension
 * @tparam K_DIM dimensions that are ignored from the state, 1 ignores the first, 2 the first and second, etc.
 * For example in AutoRally we want to input the last 4 values of our state (dim 7), so our K is 3
 * If you use the entire state in the NN K should equal 0
 * @tparam layer_args size of the NN layers
 */

using namespace MPPI_internal;

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
class NeuralNetModel : public Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM> {
public:
  // TODO remove duplication of calculation of values, pull from the structure
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Define Eigen fixed size matrices
  using state_array = typename Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>::state_array;
  using control_array = typename Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>::control_array;
  using dfdx = typename Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>::dfdx;
  using dfdu = typename Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>, NNDynamicsParams<S_DIM, C_DIM, K_DIM, layer_args...>, S_DIM, C_DIM>::dfdu;

  static const int DYNAMICS_DIM = S_DIM - K_DIM; ///< number of inputs from state
  static const int NUM_LAYERS = layer_counter(layer_args...); ///< Total number of layers (including in/out layer)
  static const int PRIME_PADDING = 1; ///< Extra padding to largest layer to avoid shared mem bank conflicts
  static const int LARGEST_LAYER = neuron_counter(layer_args...) + PRIME_PADDING; ///< Number of neurons in the largest layer(including in/out neurons)
  static const int NUM_PARAMS = param_counter(layer_args...); ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD = 0; ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK = 2*LARGEST_LAYER; ///< Amount of shared memory we need per ROLLOUT.

  NeuralNetModel(cudaStream_t stream=0);
  NeuralNetModel(std::array<float2, C_DIM> control_rngs, cudaStream_t stream=0);

  ~NeuralNetModel();

  std::array<int, NUM_LAYERS> getNetStructure() {
    std::array<int, NUM_LAYERS> array;
    for(int i = 0; i < NUM_LAYERS; i++) {
      array[i] = this->params_.net_structure[i];
    }
    return array;
  }
  std::array<int, (NUM_LAYERS - 1) * 2> getStideIdcs() {
    std::array<int, (NUM_LAYERS - 1) * 2> array;
    for(int i = 0; i < (NUM_LAYERS - 1)*2; i++) {
      array[i] = this->params_.stride_idcs[i];
    }
    return array;
  }
  std::array<float, NUM_PARAMS> getTheta() {
    std::array<float, NUM_PARAMS> array;
    for(int i = 0; i < NUM_PARAMS; i++) {
      array[i] = this->params_.theta[i];
    }
    return array;
  }
  __device__ int* getNetStructurePtr(){return this->params_.net_structure;}
  __device__ int* getStrideIdcsPtr(){return this->params_.stride_idcs;}
  __device__ float* getThetaPtr(){return this->params_.theta;}

  void CPUSetup();

  void paramsToDevice();

  void freeCudaMem();

  void printState();

  void printParams();

  void loadParams(const std::string& model_path);

  void updateModel(std::vector<int> description, std::vector<float> data);

  void computeGrad(const Eigen::Ref<const state_array>& state,
                   const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A,
                   Eigen::Ref<dfdu> B);

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
          Eigen::Ref<state_array> state_der);
  void computeKinematics(const Eigen::Ref<const state_array>& state, Eigen::Ref<state_array> s_der);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s = nullptr);
  __device__ void computeKinematics(float* state, float* state_der);
//  __device__ void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s);

private:
  Eigen::MatrixXf* weighted_in_ = nullptr;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* weights_ = nullptr;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* biases_ = nullptr;
};

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

#if __CUDACC__
#include "ar_nn_model.cu"
#endif

#endif // AR_NN_DYNAMICS_CUH_
