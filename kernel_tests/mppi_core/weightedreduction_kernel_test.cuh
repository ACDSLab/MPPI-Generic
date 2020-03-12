#pragma once

#ifndef MPPI_GENERIC_KERNEL_TESTS_MPPI_CORE_WEIGHTEDREDUCTION_KERNEL_TEST_CUH_
#define MPPI_GENERIC_KERNEL_TESTS_MPPI_CORE_WEIGHTEDREDUCTION_KERNEL_TEST_CUH_

#include <mppi_core/mppi_common.cuh>
#include <array>


__global__ void setInitialControlToZero_KernelTest(int control_dim, float* u_d, float* u_intermediate);

template<int num_threads, int control_dim>
void launchSetInitialControlToZero_KernelTest(std::array<float, control_dim>& u_host,
        std::array<float, num_threads*control_dim>& u_intermediate_host);

template<int CONTROL_DIM>
__global__ void strideControlWeightReduction_KernelTest(int num_rollouts, int num_timesteps, int sum_stride,
                                                        float* exp_costs_d, float normalizer, float* du_d, float* u_intermediate);

template<int control_dim, int num_rollouts, int num_timesteps, int sum_stride>
void launchStrideControlWeightReduction_KernelTest(float normalizer,
                                                   const std::array<float, num_rollouts>& exp_costs_host,
                                                   const std::array<float, num_rollouts*num_timesteps*control_dim>& du_host,
                                                   std::array<float, num_timesteps*control_dim*((num_rollouts - 1) / sum_stride + 1)>& u_intermediate_host);

template<int control_dim>
__global__ void rolloutWeightReductionAndSaveControl_KernelTest(int num_rollouts, int num_timesteps, int sum_stride,
                                                                float* u_intermediate, float* du_new_d);

template<int control_dim, int num_rollouts, int num_timesteps, int sum_stride>
void launchRolloutWeightReductionAndSaveControl_KernelTest(const std::array<float, num_timesteps*control_dim*((num_rollouts - 1) / sum_stride + 1)>& u_intermediate_host,
                                                           std::array<float, num_timesteps*control_dim>& du_new_host);

#if __CUDACC__
#include "weightedreduction_kernel_test.cu"
#endif

#endif //! MPPI_GENERIC_KERNEL_TESTS_MPPI_CORE_WEIGHTEDREDUCTION_KERNEL_TEST_CUH_