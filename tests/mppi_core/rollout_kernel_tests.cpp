#include <gtest/gtest.h>
#include <dynamics/cartpole/cartpole.cuh>
#include <mppi_core/mppi_common.cuh>
#include <kernel_tests/mppi_core/rollout_kernel_test.cuh>

/*
 * Here we will test various device functions that are related to cuda kernel things.
 */

TEST(RolloutKernel, Device_loadGlobalToShared) {
    launchGlobalToShared_KernelTest();
    FAIL();
}


TEST(Kernel, One) {
    // First lets instantiate a cartpole
    auto CP_host = new Cartpole(0.1, 1.0, 2.0, 3.0);
    delete(CP_host);
}

TEST(Eigen, blah) {
    FAIL();
}