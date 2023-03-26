#pragma once
/**
 * Created by Bogdan Vlahov on 3/24/2023
 **/

#include <mppi/sampling_distributions/sampling_distribution.cuh>

namespace mppi
{
namespace sampling_distributions
{
__global__ void setGaussianControls(const float* __restrict__ mean_d, const float* __restrict__ std_dev_d,
                                    float* __restrict__ control_samples_d, const int control_dim,
                                    const int num_timesteps, const int num_rollouts, const int num_distributions,
                                    const int optimization_stride, const float pure_noise_percentage,
                                    const bool time_specific_std_dev = false);

template <int C_DIM>
struct GaussianParams : public SamplingParams<C_DIM>
{
  float std_dev[C_DIM] = { 0.0f };
  dim3 rewrite_controls_block_dim(32, 32, 1);
  bool time_specific_std_dev = false;
  GaussianParams()
  {
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      std_dev[i] = 1.0f;
    }
  }
};

template <int C_DIM, int MAX_TIMESTEPS = 2>
struct GaussianTimeVaryingStdDevParams : public GaussianParams<C_DIM>
{
  float std_dev[C_DIM * MAX_TIMESTEPS] = { 0.0f };
  GaussianTimeVaryingStdDevParams()
  {
    time_specific_std_dev = true;
    for (int i = 0; i < CONTROL_DIM * MAX_TIMESTEPS; i++)
    {
      std_dev[i] = 1.0f;
    }
  }
};

template <class CLASS_T, template <int> class PARAMS_TEMPLATE = GaussianParams, class DYN_PARAMS_T = DynamicsParams>
class GaussianDistributionImpl : public SamplingDistribution<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = typename SamplingDistribution<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>;
  using SAMPLING_PARAMS_T = typename PARENT_CLASS::SAMPLING_PARAMS_T;
  static const int SHARED_MEM_REQUEST_BLK_BYTES = CONTROL_DIM * sizeof(float);  // used to hold epsilon = v - mu

  GaussianDistributionImpl(const int control_dim, const int num_rollouts, const int num_timesteps);
  GaussianDistributionImpl(const SAMPLING_PARAMS_T& params);

  __host__ void generateSamples(const int& optimization_stride, const int& iteration_num, curandGenerator_t& gen);

  __host__ void allocateCUDAMemoryHelper();

  __host__ void paramsToDevice(bool synchronize = true);

  __device__ void getControlSample(const int& sample_index, const int& t, const int& distribution_index,
                                   const float* state, float* control, float* theta_d, const int& block_size = 1,
                                   const int& thread_index = 1);

  __host__ void updateDistributionFromDevice(const float* trajectory_weights_d, float normalizer,
                                             const int& distribution_i, bool synchronize = false);

  __host__ void setHostOptimalControlSequence(float* optimal_control_trajectory, const int& distribution_idx,
                                              bool synchronize = true);

  __host__ __device__ float computeLikelihoodRatioCost(const float* u, const float* theta_d, const float lambda = 1.0,
                                                       const float alpha = 0.0);

protected:
  float* std_dev_d_ = nullptr;
  float* control_means_d_ = nullptr;
};
}  // namespace sampling_distributions
}  // namespace mppi
