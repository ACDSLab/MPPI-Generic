#ifndef MPPIGENERIC_FNN_HELPER_CUH
#define MPPIGENERIC_FNN_HELPER_CUH

#include "meta_math.h"
#include <mppi/utils/managed.cuh>
#include <mppi/utils/file_utils.h>
#include <cnpy.h>

template<int... layer_args>
struct FNNParams {
  static const int NUM_LAYERS = layer_counter(layer_args...);  ///< Total number of layers (including in/out layer)
  static const int PRIME_PADDING = 1;  ///< Extra padding to largest layer to avoid shared mem bank conflicts
  static const int LARGEST_LAYER = neuron_counter(layer_args...) +
                                   PRIME_PADDING;  ///< Number of neurons in the largest layer(including in/out neurons)
  static const int NUM_PARAMS = param_counter(layer_args...);   ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD = 0;                  ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK = 2 * LARGEST_LAYER;  ///< Amount of shared memory we need per ROLLOUT.

  static const int INPUT_DIM = input_dim(layer_args...);
  static const int OUTPUT_DIM = output_dim(layer_args...);

  typedef float ThetaArr[NUM_PARAMS];
  typedef int NetStructureArr[NUM_LAYERS];
  typedef int StrideIcsArr[(NUM_LAYERS - 1) * 2];

  // packed by all weights that connect layer 1 to layer 2 neuron 1, bias for all connections from layer 1 to layer 2
  // then layer 2 neuron 2, etc
  ThetaArr theta = { 0.0 };
  // TODO stride_idcs and net_strucutre should be write protected, so user cannot modify these values

  // index into theta for weights and bias (layer 0 weights start, no bias in input layer, layer 1 weights start, layer1
  // bias start...
  StrideIcsArr stride_idcs = { 0 };

  //[neurons in layer 1, neurons in layer 2, ...]
  NetStructureArr net_structure = { layer_args... };

  FNNParams()
  {
    int stride = 0;
    for (int i = 0; i < NUM_LAYERS - 1; i++)
    {
      stride_idcs[2 * i] = stride;
      stride += net_structure[i + 1] * net_structure[i];
      stride_idcs[2 * i + 1] = stride;
      stride += net_structure[i + 1];
    }
  }
};

template<class PARAMS_T>
class FNNHelper : public Managed
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static const int NUM_LAYERS = PARAMS_T::NUM_LAYERS;  ///< Total number of layers (including in/out layer)
  static const int PRIME_PADDING = PARAMS_T::PRIME_PADDING;  ///< Extra padding to largest layer to avoid shared mem bank conflicts
  static const int LARGEST_LAYER = PARAMS_T::LARGEST_LAYER;  ///< Number of neurons in the largest layer(including in/out neurons)
  static const int NUM_PARAMS = PARAMS_T::NUM_PARAMS;   ///< Total number of model parameters;
  static const int SHARED_MEM_REQUEST_GRD = PARAMS_T::SHARED_MEM_REQUEST_GRD; ///< Amount of shared memory we need per BLOCK.
  static const int SHARED_MEM_REQUEST_BLK = PARAMS_T::SHARED_MEM_REQUEST_BLK;  ///< Amount of shared memory we need per ROLLOUT.

  static const int INPUT_DIM = PARAMS_T::INPUT_DIM;
  static const int OUTPUT_DIM = PARAMS_T::OUTPUT_DIM;

  typedef Eigen::Matrix<float, PARAMS_T::INPUT_DIM, 1> input_array;
  typedef Eigen::Matrix<float, PARAMS_T::OUTPUT_DIM, 1> output_array;
  typedef Eigen::Matrix<float, PARAMS_T::OUTPUT_DIM, PARAMS_T::INPUT_DIM> dfdx;

  explicit FNNHelper<PARAMS_T>(cudaStream_t stream = 0);
  explicit FNNHelper<PARAMS_T>(std::string, cudaStream_t stream = 0);
  ~FNNHelper();

  void loadParams(const std::string& model_path);
  void loadParams(const cnpy::npz_t& npz);

  void CPUSetup();

  void GPUSetup();

  void updateModel(const std::vector<int>& description, const std::vector<float>& data);

  void freeCudaMem();

  void paramsToDevice();

  bool computeGrad(const Eigen::Ref<const input_array>& input,
                   Eigen::Ref<dfdx> A);

  void forward(const Eigen::Ref<const input_array>& input, Eigen::Ref<output_array> output);

  __device__ void forward(float* input, float* output, float* theta_s);

  std::array<int, NUM_LAYERS> getNetStructure()
  {
    std::array<int, NUM_LAYERS> array;
    for (int i = 0; i < NUM_LAYERS; i++)
    {
      array[i] = this->params_.net_structure[i];
    }
    return array;
  }
  std::array<int, (NUM_LAYERS - 1) * 2> getStideIdcs()
  {
    std::array<int, (NUM_LAYERS - 1) * 2> array;
    for (int i = 0; i < (NUM_LAYERS - 1) * 2; i++)
    {
      array[i] = this->params_.stride_idcs[i];
    }
    return array;
  }
  std::array<float, NUM_PARAMS> getTheta()
  {
    std::array<float, NUM_PARAMS> array;
    for (int i = 0; i < NUM_PARAMS; i++)
    {
      array[i] = this->params_.theta[i];
    }
    return array;
  }
  __device__ int* getNetStructurePtr()
  {
    return this->params_.net_structure;
  }
  __device__ int* getStrideIdcsPtr()
  {
    return this->params_.stride_idcs;
  }
  __device__ float* getThetaPtr()
  {
    return this->params_.theta;
  }

  // device pointer, null on the device
  FNNHelper<PARAMS_T>* network_d_ = nullptr;

private:
  Eigen::MatrixXf weighted_in_[NUM_LAYERS];
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights_[NUM_LAYERS];
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> biases_[NUM_LAYERS];

  // params
  PARAMS_T params_;
};

#if __CUDACC__
#include "fnn_helper.cu"
#endif

#endif  // MPPIGENERIC_FNN_HELPER_CUH
