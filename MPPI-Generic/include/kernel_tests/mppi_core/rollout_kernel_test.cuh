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

#endif // !KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_