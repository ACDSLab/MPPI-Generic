//
// Created by mgandhi on 5/23/20.
//

#ifndef MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
#define MPPIGENERIC_RMPPI_KERNEL_TEST_CUH

#include <mppi/core/rmppi_kernels.cuh>

template <class DYN_T, class COST_T, class SAMPLER_T>
void launchCPUInitEvalKernel(DYN_T* model, COST_T* cost, SAMPLER_T* sampler, const float dt, const int num_timesteps,
                             const int num_candidates, const int num_samples, const float lambda, const float alpha,
                             const Eigen::Ref<const Eigen::MatrixXf>& candidates,
                             const Eigen::Ref<const Eigen::MatrixXi>& strides,
                             Eigen::Ref<Eigen::MatrixXf> trajectory_costs);

#if __CUDACC__
#include "rmppi_kernel_test.cu"
#endif

#endif  // MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
