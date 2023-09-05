#ifndef AR_NN_DYNAMICS_CUH_
#define AR_NN_DYNAMICS_CUH_

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/file_utils.h>
#include <mppi/utils/nn_helpers/meta_math.h>
#include <mppi/utils/nn_helpers/fnn_helper.cuh>

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cnpy.h>

#include <vector>
/**
 * For AutoRally
 * State x, y, yaw, roll, u_x, u_y, yaw_rate
 * NeuralNetModel<7,2,3,6,32,32,4> model(dt, u_constraint);
 * DYNAMICS_DIM = 4
 */


struct NNDynamicsParams : public DynamicsParams
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
    NUM_STATES
  };
  enum class ControlIndex : int
  {
    STEERING = 0,
    THROTTLE,
    NUM_CONTROLS
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
    NUM_OUTPUTS
  };
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
class NeuralNetModel : public Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>,
                                       NNDynamicsParams>
{
public:
  // TODO remove duplication of calculation of values, pull from the structure
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using PARENT_CLASS = Dynamics<NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>,
                                NNDynamicsParams>;

  // Define Eigen fixed size matrices
  typedef FNNHelper<FNNParams<layer_args...>> NN;
  using state_array = typename PARENT_CLASS::state_array;
  using control_array = typename PARENT_CLASS::control_array;
  using output_array = typename PARENT_CLASS::output_array;
  using dfdx = typename PARENT_CLASS::dfdx;
  using dfdu = typename PARENT_CLASS::dfdu;
  using nn_input_array = typename NN::input_array;
  using nn_output_array = typename NN::output_array;
  using nn_dfdx = typename NN::dfdx ;


  static const int DYNAMICS_DIM = S_DIM - K_DIM;               ///< number of inputs from state
  static const int NUM_LAYERS = NN::NUM_LAYERS;  ///< Total number of layers (including in/out layer)
  static const int PRIME_PADDING = 1;  ///< Extra padding to largest layer to avoid shared mem bank conflicts
  static const int LARGEST_LAYER = NN::LARGEST_LAYER; ///< Number of neurons in the largest layer(including in/out neurons)
  static const int NUM_PARAMS = NN::NUM_PARAMS;   ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD_BYTES = NN::SHARED_MEM_REQUEST_GRD_BYTES;  ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK_BYTES = NN::SHARED_MEM_REQUEST_BLK_BYTES;  ///< Amount of shared memory we need per ROLLOUT.

  NeuralNetModel(cudaStream_t stream = 0);
  NeuralNetModel(std::array<float2, C_DIM> control_rngs, cudaStream_t stream = 0);

  ~NeuralNetModel();

  std::string getDynamicsModelName() const override
  {
    return "FCN Autorally Model";
  }

  std::array<int, NUM_LAYERS> getNetStructure()
  {
    return helper_->getNetStructure();
  }
  std::array<int, (NUM_LAYERS - 1) * 2> getStideIdcs()
  {
    return helper_->getStideIdcs();
  }
  std::array<float, NUM_PARAMS> getTheta()
  {
    return helper_->getTheta();
  }
  __device__ int* getNetStructurePtr()
  {
    return helper_->getNetStructurePtr();
  }
  __device__ int* getStrideIdcsPtr()
  {
    return helper_->getStrideIdcsPtr();
  }
  __device__ float* getThetaPtr()
  {
    return helper_->getThetaPtr();
  }

  NN* getHelperPtr() {
    return helper_;
  }

  void paramsToDevice();

  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt);

  void initializeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                          Eigen::Ref<output_array> output, float t_0, float dt)
  {
    PARENT_CLASS::initializeDynamics(state, control, output, t_0, dt);
  }

  void GPUSetup();

  void freeCudaMem();

  void printState();

  void printParams();

  void loadParams(const std::string& model_path);

  void updateModel(std::vector<int> description, std::vector<float> data);

  bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);
  void computeKinematics(const Eigen::Ref<const state_array>& state, Eigen::Ref<state_array> s_der);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s = nullptr);
  __device__ void computeKinematics(float* state, float* state_der);

  state_array stateFromMap(const std::map<std::string, float>& map) override;

private:
  NN* helper_ = nullptr;
};

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::DYNAMICS_DIM;

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NUM_LAYERS;

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::LARGEST_LAYER;

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NUM_PARAMS;

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::SHARED_MEM_REQUEST_GRD_BYTES;

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
const int NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::SHARED_MEM_REQUEST_BLK_BYTES;

#if __CUDACC__
#include "ar_nn_model.cu"
#endif

#endif  // AR_NN_DYNAMICS_CUH_
