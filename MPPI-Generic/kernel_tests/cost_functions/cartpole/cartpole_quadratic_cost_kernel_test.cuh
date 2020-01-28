#ifndef MPPIGENERIC_CARTPOLE_QUADRATIC_COST_KERNEL_TEST_CUH
#define MPPIGENERIC_CARTPOLE_QUADRATIC_COST_KERNEL_TEST_CUH

#include <cost_functions/cartpole/cartpole_quadratic_cost.cuh>

/**
 * In this cost function, we are supplied with a device side cost class, and a device side parameter structure
 */
__global__ void parameterTestKernel(CartPoleQuadraticCost* cost_d, CartPoleQuadraticCost::CartPoleQuadraticCostParams* params_d);

/**
 *
 */
void launchParameterTestKernel(const CartPoleQuadraticCost& cost, CartPoleQuadraticCost::CartPoleQuadraticCostParams& param_check);

#endif //!MPPIGENERIC_CARTPOLE_QUADRATIC_COST_KERNEL_TEST_CUH