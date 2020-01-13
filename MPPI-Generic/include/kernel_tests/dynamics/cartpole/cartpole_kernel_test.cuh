#pragma once

#ifndef KERNEL_TESTS_DYNAMICS_CARTPOLE_CARTPOLE_KERNEL_TEST_CUH_
#define KERNEL_TESTS_DYNAMICS_CARTPOLE_CARTPOLE_KERNEL_TEST_CUH_

#include <dynamics/cartpole/cartpole.cuh>

__global__ void CartMassTestKernel(Cartpole* CP, float& mass_check);
__global__ void PoleMassTestKernel(Cartpole* CP, float& mass_check);
__global__ void PoleLengthTestKernel(Cartpole* CP, float& length_check);
__global__ void GravityTestKernel(Cartpole* CP, float& gravity_check);
__global__ void DynamicsTestKernel(Cartpole* CP, float* state, float* control, float* state_der);

void launchCartMassTestKernel(const Cartpole&, float& mass_check);
void launchPoleMassTestKernel(const Cartpole&, float& mass_check);
void launchPoleLengthTestKernel(const Cartpole&, float& length_check);
void launchGravityTestKernel(const Cartpole&, float& gravity_check);
void launchDynamicsTestKernel(const Cartpole&, float* state_cpu,
                              float* control_cpu, float* state_der_cpu);

#endif // !KERNEL_TESTS_DYNAMICS_CARTPOLE_CARTPOLE_KERNEL_TEST_CUH_
