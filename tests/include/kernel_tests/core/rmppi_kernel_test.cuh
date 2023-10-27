//
// Created by mgandhi on 5/23/20.
//

#ifndef MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
#define MPPIGENERIC_RMPPI_KERNEL_TEST_CUH

#include <mppi/core/mppi_common.cuh>
#include <mppi/feedback_controllers/CCM/ccm.h>
#include <curand.h>
#include <Eigen/Dense>

template <class DYN_T, class COST_T, class FB_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelGPU(DYN_T* dynamics, COST_T* costs, FB_T* fb_controller, float dt, int num_timesteps,
                                 int optimization_stride, float lambda, float alpha, float value_func_threshold,
                                 const std::vector<float>& x0_nom, const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u, const std::vector<float>& nom_control_seq,
                                 const std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom, cudaStream_t stream = 0);

template <class DYN_T, class COST_T, class FB_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelCPU(DYN_T* dynamics, COST_T* costs, FB_T* fb_controller, float dt, int num_timesteps,
                                 int optimization_stride, float lambda, float alpha, float value_func_threshold,
                                 const std::vector<float>& x0_nom, const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u, const std::vector<float>& nom_control_seq,
                                 std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom);

template <class DYNAMICS_T, class COSTS_T, class FB_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int BLOCKSIZE_X,
          int BLOCKSIZE_Y>
void launchComparisonRolloutKernelTest(
    DYNAMICS_T* dynamics, COSTS_T* costs, FB_T* fb_controller, float dt, float lambda, float alpha,
    std::array<float, DYNAMICS_T::STATE_DIM> state_array, std::array<float, DYNAMICS_T::STATE_DIM> state_array_nominal,
    std::array<float, NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM> control_array,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM> control_noise_array,
    std::array<float, DYNAMICS_T::CONTROL_DIM> sigma_u, std::array<float, 2 * NUM_ROLLOUTS>& rmppi_costs_out,
    std::array<float, NUM_ROLLOUTS>& mppi_costs_out, int opt_delay, cudaStream_t stream);

template <class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelCCMCPU(DYN_T* model, COST_T* costs, ccm::LinearCCM<DYN_T, NUM_TIMESTEPS>* fb_controller,
                                    float dt, int num_timesteps, int optimization_stride, float lambda, float alpha,
                                    float value_func_threshold, const std::vector<float>& x0_nom,
                                    const std::vector<float>& x0_act, const std::vector<float>& sigma_u,
                                    const std::vector<float>& nom_control_seq, std::vector<float>& sampled_noise,
                                    std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                    std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom);

#if __CUDACC__
#include "rmppi_kernel_test.cu"
#endif

#endif  // MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
