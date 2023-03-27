#pragma once
/**
 * Created by Bogdan Vlahov on 3/24/2023
 **/

#include <mppi/sampling_distributions/sampling_distribution.cuh>

#include <vector>

namespace mppi
{
namespace sampling_distributions
{
__global__ void setGaussianControls(const float* __restrict__ mean_d, const float* __restrict__ std_dev_d,
                                    float* __restrict__ control_samples_d, const int control_dim,
                                    const int num_timesteps, const int num_rollouts, const int num_distributions,
                                    const int optimization_stride, const float pure_noise_percentage,
                                    const bool time_specific_std_dev = false);

// Set the default number of distributions to 2 since that is currently the most we would use
template <int C_DIM, int MAX_DISTRIBUTIONS_T = 2>
struct GaussianParams : public SamplingParams<C_DIM>
{
  static const int MAX_DISTRIBUTIONS = MAX_DISTRIBUTIONS_T;
  float std_dev[C_DIM * MAX_DISTRIBUTIONS] = { 0.0f };
  float control_cost_coeff[C_DIM] = { 0.0f };
  float pure_noise_trajectories_percentage = 0.01f;
  // Kernel launching params
  dim3 rewrite_controls_block_dim(32, 32, 1);
  int sum_strides = 32;
  // Various flags
  bool time_specific_std_dev = false;
  bool use_same_noise_for_all_distributions = true;

  GaussianParams(int num_rollouts = 1, int num_timesteps = 1, int num_distributions = 1)
    : SamplingParams<C_DIM>::SamplingParams(num_rollouts, num_timesteps, num_distributions)
  {
    for (int i = 0; i < CONTROL_DIM * MAX_DISTRIBUTIONS; i++)
    {
      std_dev[i] = 1.0f;
    }
  }

  void copyStdDevToDistribution(const int src_distribution_idx, const int dest_distribution_idx)
  {
    bool src_out_of_distribution = src_out_of_distribution >= MAX_DISTRIBUTIONS;
    if (src_out_of_distribution || dest_distribution_idx >= MAX_DISTRIBUTIONS)
    {
      printf("%s Distribution %d is out of range. There are only %d total distributions\n",
             src_out_of_distribution ? "Src" : "Dest",
             src_out_of_distribution ? src_distribution_idx : dest_distribution_idx, MAX_DISTRIBUTIONS);
      return;
    }
    float* std_dev_src = std_dev[CONTROL_DIM * src_distribution_idx];
    float* std_dev_dest = std_dev[CONTROL_DIM * dest_distribution_idx];
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      std_dev_dest[i] = std_dev_src[i];
    }
  }
};

template <int C_DIM, int MAX_TIMESTEPS = 2, int MAX_DISTRIBUTIONS_T = 2>
struct GaussianTimeVaryingStdDevParams : public GaussianParams<C_DIM, MAX_DISTRIBUTIONS_T>
{
  float std_dev[C_DIM * MAX_TIMESTEPS * MAX_DISTRIBUTIONS_T] = { 0.0f };
  GaussianTimeVaryingStdDevParams(int num_rollouts = 1, int num_timesteps = 1, int num_distributions = 1)
    : GaussianParams<C_DIM>::GaussianParams(num_rollouts, num_timesteps, num_distributions)
  {
    time_specific_std_dev = true;
    for (int i = 0; i < CONTROL_DIM * MAX_TIMESTEPS * MAX_DISTRIBUTIONS; i++)
    {
      std_dev[i] = 1.0f;
    }
  }

  void copyStdDevToDistribution(const int src_distribution_idx, const int dest_distribution_idx)
  {
    bool src_out_of_distribution = src_out_of_distribution >= MAX_DISTRIBUTIONS;
    if (src_out_of_distribution || dest_distribution_idx >= MAX_DISTRIBUTIONS)
    {
      printf("%s Distribution %d is out of range. There are only %d total distributions\n",
             src_out_of_distribution ? "Src" : "Dest",
             src_out_of_distribution ? src_distribution_idx : dest_distribution_idx, MAX_DISTRIBUTIONS);
      return;
    }
    float* std_dev_src = std_dev[CONTROL_DIM * this->num_timesteps * src_distribution_idx];
    float* std_dev_dest = std_dev[CONTROL_DIM * this->num_timesteps * dest_distribution_idx];
    for (int i = 0; i < CONTROL_DIM * this->num_timesteps; i++)
    {
      std_dev_dest[i] = std_dev_src[i];
    }
  }
};

template <class CLASS_T, template <int> class PARAMS_TEMPLATE = GaussianParams, class DYN_PARAMS_T = DynamicsParams>
class GaussianDistributionImpl : public SamplingDistribution<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = typename SamplingDistribution<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>;
  using SAMPLING_PARAMS_T = typename PARENT_CLASS::SAMPLING_PARAMS_T;
  using control_array = typename PARENT_CLASS::control_array;

  GaussianDistributionImpl(cudaStream_t stream = 0);
  GaussianDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream = 0);

  __host__ std::string getSamplingDistributionName()
  {
    return "Gaussian";
  }

  __host__ void allocateCUDAMemoryHelper();

  __host__ __device__ float computeLikelihoodRatioCost(const float* __restrict__ u, const float* __restrict__ theta_d,
                                                       const int t, const int distribution_idx,
                                                       const float lambda = 1.0, const float alpha = 0.0);

  __host__ float computeLikelihoodRatioCost(const Eigen::Ref<const control_array>& u, const int t,
                                            const int distribution_idx, const float lambda = 1.0,
                                            const float alpha = 0.0);

  __host__ void freeCudaMem();

  __host__ void generateSamples(const int& optimization_stride, const int& iteration_num, curandGenerator_t& gen,
                                bool synchronize = true);

  __host__ void paramsToDevice(bool synchronize = true);

  __host__ void setHostOptimalControlSequence(float* optimal_control_trajectory, const int& distribution_idx,
                                              bool synchronize = true);

  __host__ void updateDistributionParamsFromDevice(const float* trajectory_weights_d, float normalizer,
                                                   const int& distribution_i, bool synchronize = false);

protected:
  float* std_dev_d_ = nullptr;
  float* control_means_d_ = nullptr;
  std::vector<float> means_;
};

template <class DYN_PARAMS_T>
class GaussianDistribution : public GaussianDistributionImpl<GaussianDistribution, GaussianParams, DYN_PARAMS_T>
{
  using PARENT_CLASS = GaussianDistributionImpl<GaussianDistribution, GaussianParams, DYN_PARAMS_T>;
  using SAMPLING_PARAMS_T = PARENT_CLASS::SAMPLING_PARAMS_T;

  GaussianDistribution(cudaStream_t stream = 0) : PARENT_CLASS(stream)
  {
  }
  GaussianDistribution(const SAMPLING_PARAMS_T& params, cudaStream_t stream = 0) : PARENT_CLASS(params, stream)
  {
  }
};

#if __CUDACC__
#include "gaussian.cu"
#endif
}  // namespace sampling_distributions
}  // namespace mppi
