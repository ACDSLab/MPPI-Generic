#pragma once

#ifndef KERNEL_TESTS_MPPI_CORE_MPPI_CORE_KERNEL_TEST_CUH_
#define KERNEL_TESTS_MPPI_CORE_MPPI_CORE_KERNEL_TEST_CUH_

#include <mppi/core/mppi_common.cuh>
#include <curand.h>
#include <vector>
#include <array>

// Declare some sizes for the kernel parameters

template <int BLOCKSIZE_Z = 1>
__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device, float* x_thread,
                                              float* xdot_thread, float* u_thread, float* du_thread,
                                              float* sigma_u_thread);

void launchGlobalToShared_KernelTest(const std::vector<float>& x0_host, const std::vector<float>& u_var_host,
                                     std::vector<float>& x_thread_host, std::vector<float>& xdot_thread_host,
                                     std::vector<float>& u_thread_host, std::vector<float>& du_thread_host,
                                     std::vector<float>& sigma_u_thread_host);

void launchGlobalToShared_KernelTest_nom_act(
    const std::vector<float>& x0_host_act, const std::vector<float>& u_var_host, std::vector<float>& x_thread_host_act,
    std::vector<float>& xdot_thread_host_act, std::vector<float>& u_thread_host_act,
    std::vector<float>& du_thread_host_act, const std::vector<float>& x0_host_nom,
    std::vector<float>& x_thread_host_nom, std::vector<float>& xdot_thread_host_nom,
    std::vector<float>& u_thread_host_nom, std::vector<float>& du_thread_host_nom,
    std::vector<float>& sigma_u_thread_host);

template <class DYN_T, class COST_T, int NUM_ROLLOUTS>
void launchRolloutKernel_nom_act(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, float lambda, float alpha,
                                 const std::vector<float>& x0, const std::vector<float>& sigma_u,
                                 const std::vector<float>& nom_control_seq, std::vector<float>& trajectory_costs_act,
                                 std::vector<float>& trajectory_costs_nom, cudaStream_t stream = 0);

__global__ void injectControlNoiseOnce_KernelTest(int num_rollouts, int num_timesteps, int timestep,
                                                  float* u_traj_device, float* ep_v_device, float* sigma_u_device,
                                                  float* control_compute_device);

void launchInjectControlNoiseOnce_KernelTest(const std::vector<float>& u_traj_host, const int num_rollouts,
                                             const int num_timesteps, std::vector<float>& ep_v_host,
                                             std::vector<float>& sigma_u_host, std::vector<float>& control_compute);

template <int control_dim, int blocksize_y>
__global__ void injectControlNoiseCheckControlV_KernelTest(int num_rollouts, int num_timesteps, int timestep,
                                                           float* u_traj_device, float* ep_v_device,
                                                           float* sigma_u_device);

template <int num_rollouts, int num_timesteps, int control_dim, int blocksize_x, int blocksize_y, int gridsize_x>
void launchInjectControlNoiseCheckControlV_KernelTest(
    const std::array<float, num_timesteps * control_dim>& u_traj_host,
    std::array<float, num_rollouts * num_timesteps * control_dim>& ep_v_host,
    const std::array<float, control_dim>& sigma_u_host);

template <class COST_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int STATE_DIM, int CONTROL_DIM>
__global__ void computeRunningCostAllRollouts_KernelTest(COST_T* cost_d, float dt, float* x_trajectory_d,
                                                         float* u_trajectory_d, float* du_trajectory_d, float* var_d,
                                                         float* cost_allrollouts_d);

template <class COST_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int STATE_DIM, int CONTROL_DIM>
void computeRunningCostAllRollouts_CPU_TEST(
    COST_T& cost, float dt, std::array<float, STATE_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS>& x_trajectory,
    std::array<float, CONTROL_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS>& u_trajectory,
    std::array<float, CONTROL_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS>& du_trajectory,
    std::array<float, CONTROL_DIM>& sigma_u, std::array<float, NUM_ROLLOUTS>& cost_allrollouts);

template <class COST_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int STATE_DIM, int CONTROL_DIM>
void launchComputeRunningCostAllRollouts_KernelTest(
    const COST_T& cost, float dt, const std::array<float, STATE_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS>& x_trajectory,
    const std::array<float, CONTROL_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS>& u_trajectory,
    const std::array<float, CONTROL_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS>& du_trajectory,
    const std::array<float, CONTROL_DIM>& sigma_u, std::array<float, NUM_ROLLOUTS>& cost_allrollouts);

template <class DYN_T, int NUM_ROLLOUTS, int BLOCKSIZE_X>
__global__ void computeStateDerivAllRollouts_KernelTest(DYN_T* dynamics_d, float* x_trajectory_d, float* u_trajectory_d,
                                                        float* xdot_trajectory_d);

template <class DYN_T, int NUM_ROLLOUTS, int BLOCKSIZE_X>
void launchComputeStateDerivAllRollouts_KernelTest(
    const DYN_T& dynamics, const std::array<float, DYN_T::STATE_DIM * NUM_ROLLOUTS>& x_trajectory,
    const std::array<float, DYN_T::CONTROL_DIM * NUM_ROLLOUTS>& u_trajectory,
    std::array<float, DYN_T::STATE_DIM * NUM_ROLLOUTS>& xdot_trajectory);

template <class DYN_T>
__global__ void incrementStateAllRollouts_KernelTest(DYN_T* dynamics, int state_dim, int num_rollouts, float dt,
                                                     float* x_trajectory_d, float* xdot_trajectory_d);

template <class DYN_T, int STATE_DIM, int NUM_ROLLOUTS>
void launchIncrementStateAllRollouts_KernelTest(const DYN_T& dynamics, float dt,
                                                std::array<float, STATE_DIM * NUM_ROLLOUTS>& x_traj,
                                                std::array<float, STATE_DIM * NUM_ROLLOUTS>& xdot_traj);

template <class COST_T>
__global__ void computeAndSaveCostAllRollouts_KernelTest(COST_T* cost, int state_dim, int num_rollouts,
                                                         float* running_costs, float* terminal_state,
                                                         float* cost_rollout_device);

template <class COST_T, int STATE_DIM, int NUM_ROLLOUTS>
void launchComputeAndSaveCostAllRollouts_KernelTest(COST_T& cost,
                                                    const std::array<float, NUM_ROLLOUTS>& cost_all_rollouts,
                                                    const std::array<float, STATE_DIM * NUM_ROLLOUTS>& terminal_states,
                                                    std::array<float, NUM_ROLLOUTS>& cost_compute);

template <class DYNAMICS_T, class COSTS_T, int NUM_ROLLOUTS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
__global__ void autorallyRolloutKernel(int num_timesteps, float* state_d, float* U_d, float* du_d, float* nu_d,
                                       float* costs_d, DYNAMICS_T* dynamics_model, COSTS_T* mppi_costs, int opt_delay,
                                       float lambda, float alpha, float dt);

template <class DYNAMICS_T, class COSTS_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
void launchAutorallyRolloutKernelTest(
    DYNAMICS_T* dynamics, COSTS_T* costs, float dt, float lambda, float alpha,
    std::array<float, DYNAMICS_T::STATE_DIM> state_array,
    std::array<float, NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM> control_array,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM> control_noise_array,
    std::array<float, DYNAMICS_T::CONTROL_DIM> sigma_u, std::array<float, NUM_ROLLOUTS>& costs_out,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM>& control_noise_out, int opt_delay,
    cudaStream_t stream);

template <class DYNAMICS_T, class COSTS_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
void launchGenericRolloutKernelTest(
    DYNAMICS_T* dynamics, COSTS_T* costs, float dt, float lambda, float alpha,
    std::array<float, DYNAMICS_T::STATE_DIM> state_array,
    std::array<float, NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM> control_array,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM> control_noise_array,
    std::array<float, DYNAMICS_T::CONTROL_DIM> sigma_u, std::array<float, NUM_ROLLOUTS>& costs_out,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM>& control_noise_out, int opt_delay,
    cudaStream_t stream);

#if __CUDACC__
#include "rollout_kernel_test.cu"
#endif

#endif  // !KERNEL_TESTS_MPPI_CORE_MPPI_CORE_KERNEL_TEST_CUH_
