//
// Created by mgandhi on 5/23/20.
//

#ifndef MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
#define MPPIGENERIC_RMPPI_KERNEL_TEST_CUH

#include <mppi/core/mppi_common.cuh>
#include <curand.h>
#include <Eigen/Dense>

template<class DYN_T, class COST_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelGPU(DYN_T* dynamics, COST_T* costs,
                                 float dt,
                                 int num_timesteps,
                                 float lambda,
                                 float value_func_threshold,
                                 const std::vector<float>& x0_nom,
                                 const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u,
                                 const std::vector<float>& nom_control_seq,
                                 const std::vector<float>& feedback_gains_seq,
                                 const std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom,
                                 cudaStream_t stream = 0);

template<class DYN_T, class COST_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelCPU(DYN_T* dynamics, COST_T* costs,
                                 float dt,
                                 int num_timesteps,
                                 float lambda,
                                 float value_func_threshold,
                                 const std::vector<float>& x0_nom,
                                 const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u,
                                 const std::vector<float>& nom_control_seq,
                                 const std::vector<float>& feedback_gains_seq,
                                 const std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom);

#if __CUDACC__
#include "rmppi_kernel_test.cu"
#endif

#endif //MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
