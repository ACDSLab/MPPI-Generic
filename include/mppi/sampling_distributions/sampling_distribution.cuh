#pragma once
/*
Created by Bogdan Vlahov on 3/22/2023
*/

#include <mppi/utils/managed.cuh>
#include <mppi/dynamics/dynamics.cuh>

#include <string>

namespace mppi
{
namespace sampling_distributions
{
// Auto-align all Parameter structures to float4 so that any data after it in saved memory has full memory alignment
template <int C_DIM>
struct alignas(float4) SamplingParams
{
  static const int CONTROL_DIM = C_DIM;
  int num_rollouts = 1;
  int num_timesteps = 1;
  int num_distributions = 1;
};

template <class CLASS_T, template <int> class PARAMS_TEMPLATE, class DYN_PARAMS_T = DynamicsParams>
class SamplingDistribution : public Managed
{
public:
  /**
   * Setup typedefs and aliases
   */
  typedef CLASS_T SAMPLING_T;
  using ControlIndex = typename DYN_PARAMS_T::ControlIndex;
  using OutputIndex = typename DYN_PARAMS_T::OutputIndex;
  using TEMPLATED_DYN_PARAMS = DYN_PARAMS_T;

  static const int CONTROL_DIM = E_INDEX(ControlIndex, NUM_CONTROLS);
  typedef PARAMS_TEMPLATE<CONTROL_DIM> SAMPLING_PARAMS_T;
  typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array;

  static_assert(std::is_base_of<SamplingParams<CONTROL_DIM>, SAMPLING_PARAMS_T>::value,
                "Sampling Distribution PARAMS_T does not inherit from SamplingParams");

  // Shared memory requests in bytes
  static const int SHARED_MEM_REQUEST_GRD_BYTES = sizeof(SAMPLING_PARAMS_T);
  static const int SHARED_MEM_REQUEST_BLK_BYTES = 0;

  SamplingDistribution(const int control_dim, const int num_samples, const int num_timesteps);
  SamplingDistribution(const SAMPLING_PARAMS_T& params);

  virtual ~SamplingDistribution()
  {
    freeCudaMem();
  }

  __host__ std::string getSamplingDistributionName()
  {
    return "Sampling distribution name not set";
  }

  void setStream(cudaStream_t stream)
  {
    stream_ = stream;
  }

  void GPUSetup();

  /**
   * deallocates the allocated cuda memory for an object
   */
  void freeCudaMem();

  /**
   * Updates the sampling distribution parameters
   * @param params
   */
  void setParams(const SAMPLING_PARAMS_T& params, bool synchronize = true)
  {
    bool reallocate_memory = params_.num_timesteps != params.num_timesteps ||
                             params_.num_rollouts != params.num_rollouts ||
                             params_.num_distributions != params.num_distributions;
    params_ = params;
    if (GPUMemStatus_)
    {
      if (reallocate_memory)
      {
        allocateCUDAMemory(false);
      }
      CLASS_T& derived = static_cast<CLASS_T&>(*this);
      derived.paramsToDevice(synchronize);
    }
  }

  __host__ __device__ SAMPLING_PARAMS_T getParams() const
  {
    return params_;
  }

  __host__ __device__ const SAMPLING_PARAMS_T getParams() const
  {
    return params_;
  }

  __host__ __device__ int getNumTimesteps() const
  {
    return this->params_.num_timesteps;
  }

  __host__ __device__ int getNumRollouts() const
  {
    return this->params_.num_rollouts;
  }

  __host__ __device__ int getNumDistributions() const
  {
    return this->params_.num_distributions;
  }

  void paramsToDevice(bool synchronize = true);
  /**
   * @brief Host call to generate the samples needed for sampling. When this method finishes running,
   * the GPU Memory should be filled with the required control samples
   *
   * @param num_timesteps - time horizon to generate samples for
   * @param num_rollouts - number of sample trajectories to create
   * @param optimization_stride - at which point along the trajectory to start sampling from
   * @param iteration_num - which iteration of the optimization you are on
   */
  __host__ void generateSamples(const int& optimization_stride, const int& iteration_num, const int& num_timesteps,
                                const int& num_rollouts, const int& num_distributions, curandGenerator_t& gen);
  __host__ void generateSamples(const int& optimization_stride, const int& iteration_num, curandGenerator_t& gen);
  __host__ void allocateCUDAMemory(bool synchronize = false);
  __host__ void allocateCUDAMemoryHelper();

  __device__ void initializeDistributions(const float* state, const float t_0, const float dt, float* theta_d);

  /**
   * @brief Get the Control Sample for control sample sample_index at time t and put it into the control array
   *
   * @param sample_index
   * @param t
   * @param control
   * @param block_size - amount of parallelization
   * @param thread_index - parallelization index
   */
  __device__ void getControlSample(const int& sample_index, const int& t, const int& distribution_index,
                                   const float* state, float* control, float* theta_d, const int& block_size = 1,
                                   const int& thread_index = 1);

  // takes in the cost of each sample generated and conducts an update of the distribution (For Gaussians, mean update)
  __host__ void updateDistributionFromDevice(const float* trajectory_weights_d, float normalizer,
                                             const int& distribution_i, bool synchronize = false);

  __host__ void updateDistributionFromHost(const Eigen::Ref<const Eigen::MatrixXf>& trajectory_weights,
                                           float normalizer, const int& distribution_i, bool synchronize = false);

  // Set a host side pointer to the optimal control sequence from the distribution
  __host__ void setHostOptimalControlSequence(float* optimal_control_trajectory, const int& distribution_idx,
                                              bool synchronize = true);

  __host__ __device__ float computeLikelihoodRatioCost(const float* u, const float* theta_d, const int t,
                                                       const float lambda = 1.0, const float alpha = 0.0);

  CLASS_T* sampling_d_ = nullptr;

protected:
  SAMPLING_PARAMS_T params_;
  float* control_samples_d_ = nullptr;
};

#if __CUDACC__
#include "sampling_distribution.cu"
#endif
}  // namespace sampling_distributions
}  // namespace mppi
