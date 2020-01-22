#pragma once

#ifndef KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_
#define KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_

#include <mppi_core/mppi_common.cuh>
#include <vector>

__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device,
                                              float* x_thread, float* xdot_thread, float* u_thread, float* du_thread, float* sigma_u_thread);

void launchGlobalToShared_KernelTest(const std::vector<float>& x0_host,const std::vector<float>& u_var_host,
                                     std::vector<float>& x_thread_host, std::vector<float>& xdot_thread_host,
                                     std::vector<float>& u_thread_host, std::vector<float>& du_thread_host, std::vector<float>& sigma_u_thread_host );

__global__ void injectControlNoiseOnce_KernelTest(float* u_traj_device, float* ep_v_device, float* control_compute_device);

void launchInjectControlNoiseOnce_KernelTest(const std::vector<float>& u_traj_host, const int num_rollouts, const int num_timesteps,
                                             std::vector<float>& ep_v_host, std::vector<float>& sigma_u_host, std::vector<float>& control_compute);

__global__ void injectControlNoise_KernelTest();

void launchInjectControlNoise_KernelTest();

#endif // !KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_