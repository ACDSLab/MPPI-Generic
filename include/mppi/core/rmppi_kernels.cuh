/**
 * Created by Bogdan Vlahov on 3/25/2023
 **/
#pragma once
#include <mppi/utils/math_utils.h>
#include <mppi/dynamics/dynamics.cuh>
#include <mppi/cost_functions/cost.cuh>

namespace mppi
{
namespace kernels
{
namespace rmppi
{
/**
 * Kernels Methods
 **/
template <class DYN_T, class SAMPLING_T>
__global__ void initEvalDynKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling, float dt,
                                  const int num_timesteps, const int num_rollouts, int samples_per_condition,
                                  const int* __restrict__ strides_d, const float* __restrict__ states_d,
                                  float* __restrict__ y_d);

template <class COST_T, class SAMPLING_T, int BLOCKSIZE_X, bool COALESCE = false>
__global__ void initEvalCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                   const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                   const int samples_per_condition, const int* __restrict__ strides_d,
                                   const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d);

template <class DYN_T, class FB_T, class SAMPLING_T, int NOMINAL_STATE_IDX = 0>
__global__ void rolloutRMPPIDynamicsKernel(DYN_T* __restrict__ dynamics, FB_T* __restrict__ fb_controller,
                                           SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                                           const int num_rollouts, const float* __restrict__ init_x_d,
                                           float* __restrict__ y_d);

template <class COST_T, class DYN_T, class SAMPLING_T, class FB_T, int NOMINAL_STATE_IDX = 0, bool COALESCE = false>
__global__ void rolloutRMPPICostKernel(COST_T* __restrict__ costs, DYN_T* __restrict__ dynamics,
                                       FB_T* __restrict__ fb_controller, SAMPLING_T* __restrict__ sampling, float dt,
                                       const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                       float value_func_threshold, const float* __restrict__ init_x_d,
                                       const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d);
/**
 * Launch Methods
 **/
template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchFastInitEvalKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs,
                              SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                              const int num_rollouts, float lambda, float alpha, int samples_per_condition,
                              int* __restrict__ strides_d, float* __restrict__ init_x_d, float* __restrict__ y_d,
                              float* __restrict__ trajectory_costs, dim3 dimDynBlock, dim3 dimCostBlock,
                              cudaStream_t stream, bool synchronize);

template <class DYN_T, class COST_T, class SAMPLING_T, class FB_T, int NOMINAL_STATE_IDX = 0>
void launchFastRMPPIRolloutKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs,
                                  SAMPLING_T* __restrict__ sampling, FB_T* __restrict__ fb_controller, float dt,
                                  const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                  float value_func_threshold, float* __restrict__ init_x_d, float* __restrict__ y_d,
                                  float* __restrict__ trajectory_costs, dim3 dimDynBlock, dim3 dimCostBlock,
                                  cudaStream_t stream, bool synchronize);
}  // namespace rmppi
}  // namespace kernels
}  // namespace mppi

#if __CUDACC__
#include "rmppi_kernels.cu"
#endif
