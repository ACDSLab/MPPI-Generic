#include <gtest/gtest.h>
#include <dynamics/cartpole/cartpole.cuh>
#include <mppi_core/mppi_common.cuh>
#include <kernel_tests/mppi_core/rollout_kernel_test.cuh>

/*
 * Here we will test various device functions that are related to cuda kernel things.
 */

TEST(RolloutKernel, Device_loadGlobalToShared) {
    std::vector<float> x0_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<float> u_var_host = {0.8, 0.9, 1.0};

    std::vector<float> x_thread_host(mppi_common::state_dim, 0.f);
    std::vector<float> xdot_thread_host(mppi_common::state_dim, 2.f);

    std::vector<float> u_thread_host(mppi_common::control_dim, 3.f);
    std::vector<float> du_thread_host(mppi_common::control_dim, 4.f);
    std::vector<float> sigma_u_thread_host(mppi_common::control_dim, 0.f);

    launchGlobalToShared_KernelTest(x0_host, u_var_host, x_thread_host, xdot_thread_host, u_thread_host, du_thread_host, sigma_u_thread_host);

    for (int i = 0; i < mppi_common::state_dim; i++) {
        EXPECT_FLOAT_EQ(x0_host[i], x_thread_host[i]) << "Failed at index: " << i;
    }

    for (int i = 0; i < mppi_common::state_dim; i++) {
        EXPECT_FLOAT_EQ(xdot_thread_host[i], 0.f) << "Failed at index: " << i;
    }

    for (int i = 0; i < mppi_common::control_dim; i++) {
        EXPECT_FLOAT_EQ(u_thread_host[i], 0.f) << "Failed at index: " << i;
    }

    for (int i = 0; i < mppi_common::control_dim; i++) {
        EXPECT_FLOAT_EQ(du_thread_host[i], 0.f) << "Failed at index: " << i;
    }

    for (int i = 0; i < mppi_common::control_dim; i++) {
        EXPECT_FLOAT_EQ(sigma_u_thread_host[i], u_var_host[i]) << "Failed at index: " << i;
    }

}