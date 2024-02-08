#ifndef MPPIGENERIC_FNN_HELPER_CUH
#define MPPIGENERIC_FNN_HELPER_CUH

#include "meta_math.h"
#include <mppi/utils/managed.cuh>
#include <mppi/utils/file_utils.h>
#include <cnpy.h>
#include <atomic>

template <bool USE_SHARED = true>
class FNNHelper : public Managed
{
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit FNNHelper<USE_SHARED>(const std::vector<int>& layers, cudaStream_t stream = 0);
  explicit FNNHelper<USE_SHARED>(std::string, cudaStream_t stream = 0);
  explicit FNNHelper<USE_SHARED>(const cnpy::npz_t& param_dict, cudaStream_t stream = 0);
  explicit FNNHelper<USE_SHARED>(const cnpy::npz_t& param_dict, std::string prefix, cudaStream_t stream = 0);
  ~FNNHelper();

  void loadParams(const std::string& model_path);
  void loadParams(const cnpy::npz_t& npz);
  void loadParams(std::string prefix, const cnpy::npz_t& npz, bool add_slash = true);

  void GPUSetup();

  void updateModel(const std::vector<int>& description, const std::vector<float>& data);
  void updateModel(const std::vector<float>& data);

  void freeCudaMem();

  void paramsToDevice();

  bool computeGrad(Eigen::Ref<Eigen::MatrixXf> A);

  bool computeGrad(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> A);

  void forward(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output);

  __device__ float* initialize(float* theta_s);

  __device__ float* forward(float* input, float* theta_s);
  __device__ float* forward(float* input, float* theta_s, float* curr_act);

  __host__ __device__ int* getNetStructurePtr()
  {
    return net_structure_;
  }
  __host__ __device__ int* getStrideIdcsPtr()
  {
    return stride_idcs_;
  }
  __host__ __device__ float* getThetaPtr()
  {
    return theta_;
  }

  __host__ __device__ int getNumLayers()
  {
    return this->NUM_LAYERS;
  }
  __host__ __device__ int getNumParams()
  {
    return this->NUM_PARAMS;
  }
  __host__ __device__ int getLargestLayer()
  {
    return this->LARGEST_LAYER;
  }
  __host__ __device__ int getStrideSize()
  {
    return this->STRIDE_SIZE;
  }

  Eigen::VectorXf getInputVector()
  {
    return Eigen::VectorXf(INPUT_DIM);
  }
  Eigen::VectorXf getOutputVector()
  {
    return Eigen::VectorXf(OUTPUT_DIM);
  }
  Eigen::MatrixXf getJacobianMatrix()
  {
    return Eigen::MatrixXf(OUTPUT_DIM, INPUT_DIM);
  }

  __device__ __host__ int getInputDim()
  {
    return INPUT_DIM;
  }
  __host__ __device__ int getOutputDim()
  {
    return OUTPUT_DIM;
  }

  __device__ float* getInputLocation(float* theta_s);

  void setAllWeights(float input)
  {
    std::vector<float> vals(NUM_PARAMS);
    std::fill(vals.begin(), vals.end(), input);
    updateModel(vals);
  }

  // device pointer, null on the device
  FNNHelper<USE_SHARED>* network_d_ = nullptr;

  float* weights_d_ = nullptr;

private:
  int NUM_LAYERS = 0;     ///< Total number of layers (including in/out layer)
  int LARGEST_LAYER = 0;  ///< Number of neurons in the largest layer(including in/out neurons)
  int NUM_PARAMS = 0;     ///< Total number of model parameters;
  int PARAM_SIZE = 0;     ///< Maximum size of the parameters stored in shared memory
  int STRIDE_SIZE = 0;

  int INPUT_DIM = 0;
  int OUTPUT_DIM = 0;

  // packed by all weights that connect layer 1 to layer 2 neuron 1, bias for all connections from layer 1 to layer 2
  // then layer 2 neuron 2, etc
  float* theta_ = nullptr;

  // index into theta for weights and bias (layer 0 weights start, no bias in input layer, layer 1 weights start, layer1
  // bias start...
  int* stride_idcs_ = nullptr;

  //[neurons in layer 1, neurons in layer 2, ...]
  int* net_structure_ = nullptr;

  std::atomic<bool> changed_weights_ = { false };

  std::vector<Eigen::MatrixXf> weighted_in_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weights_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> biases_;

  void setupMemory(std::vector<int> layers);
};

#if __CUDACC__
#include "fnn_helper.cu"
#endif

#endif  // MPPIGENERIC_FNN_HELPER_CUH
