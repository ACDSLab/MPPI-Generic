#pragma once

#ifndef KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_
#define KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_

#include <mppi_core/mppi_common.cuh>

__global__ void loadGlobalToShared_KernelTest();

void launchGlobalToShared_KernelTest();

#endif // !KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_