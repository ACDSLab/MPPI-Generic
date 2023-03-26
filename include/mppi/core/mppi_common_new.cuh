/**
 * Created by Bogdan Vlahov on 3/25/2023
 **/
#pragma once

namespace mppi
{
namespace kernels
{
/**
 * Kernels Methods
 **/
__global__ void weightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                        float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                        const int num_rollouts, const int sum_stride, const int control_dim);


/**
 * Device-only Kernel Helper Methods
 **/

__device__ void setInitialControlToZero(int control_dim, int thread_idx, float* __restrict__ u, float* __restrict__ u_intermediate);

__device__ void strideControlWeightReduction(const int num_rollouts, const int num_timesteps, const int sum_stride, const int thread_idx,
                                             const int block_idx, const int control_dim, const float* __restrict__ exp_costs_d, const float normalizer,
                                             const float* __restritct__ du_d, float* __restrict__  u, float* __restrict__ u_intermediate)
{
/**
 * Launch Kernel Methods
 **/

void launchWeightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                   float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                   const int num_rollouts, const int sum_stride, const int control_dim,
                                   cudaStream_t stream, bool synchronize);

#if __CUDACC__
#include "mppi_common_new.cu"
#endif
}
}
