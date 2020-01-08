#pragma once

#ifndef KERNEL_TESTS_DYNAMICS_CARTPOLE_CARTPOLE_KERNEL_TEST_CUH_
#define KERNEL_TESTS_DYNAMICS_CARTPOLE_CARTPOLE_KERNEL_TEST_CUH_

__global__ void ParameterTestKernel(Cartpole* CP)

void launchParameterTestKernel(const Cartpole& CP)

#endif // !KERNEL_TESTS_DYNAMICS_CARTPOLE_CARTPOLE_KERNEL_TEST_CUH_