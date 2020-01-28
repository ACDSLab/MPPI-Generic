#include <gtest/gtest.h>
#include <dynamics/cartpole/cartpole.cuh>
#include <cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi_core/mppi_common.cuh>
#include <mppi_core/rollout_kernel_test.cuh>
#include <utils/test_helper.h>

/*
 * Here we will test various device functions that are related to cuda kernel things.
 */

TEST(RolloutKernel, loadGlobalToShared) {
    std::vector<float> x0_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<float> u_var_host = {0.8, 0.9, 1.0};

    std::vector<float> x_thread_host(STATE_DIM, 0.f);
    std::vector<float> xdot_thread_host(STATE_DIM, 2.f);

    std::vector<float> u_thread_host(CONTROL_DIM, 3.f);
    std::vector<float> du_thread_host(CONTROL_DIM, 4.f);
    std::vector<float> sigma_u_thread_host(CONTROL_DIM, 0.f);

    launchGlobalToShared_KernelTest(x0_host, u_var_host, x_thread_host, xdot_thread_host, u_thread_host, du_thread_host, sigma_u_thread_host);

    array_expect_float_eq(x0_host, x_thread_host, STATE_DIM);
    array_expect_float_eq(0.f, xdot_thread_host, STATE_DIM);
    array_expect_float_eq(0.f, u_thread_host, CONTROL_DIM);
    array_expect_float_eq(0.f, du_thread_host, CONTROL_DIM);
    array_expect_float_eq(sigma_u_thread_host, u_var_host, CONTROL_DIM);
}

TEST(RolloutKernel, injectControlNoiseOnce) {
    int num_timesteps = 1;
    int num_rollouts = NUM_ROLLOUTS;
    std::vector<float> u_traj_host = {3.f, 4.f, 5.f};

    // Control noise
    std::vector<float> ep_v_host(num_rollouts*num_timesteps*CONTROL_DIM, 0.f);

    // Control at timestep 1 for all rollouts
    std::vector<float> control_compute(num_rollouts*CONTROL_DIM, 0.f);

    // Control variance for each control channel
    std::vector<float> sigma_u_host = {0.1f, 0.2f, 0.3f};

    launchInjectControlNoiseOnce_KernelTest(u_traj_host, num_rollouts, num_timesteps, ep_v_host, sigma_u_host, control_compute);

    // Make sure the first control is undisturbed
    int timestep = 0;
    int rollout = 0;
    for (int i = 0; i < CONTROL_DIM; ++i) {
        ASSERT_FLOAT_EQ(u_traj_host[i], control_compute[rollout*num_timesteps*CONTROL_DIM + timestep * CONTROL_DIM + i]);
    }

    // Make sure the last 99 percent are zero control with noise
    for (int j = num_rollouts*.99; j < num_rollouts; ++j) {
        for (int i = 0; i < CONTROL_DIM; ++i) {
            ASSERT_FLOAT_EQ(ep_v_host[j*num_timesteps*CONTROL_DIM + timestep * CONTROL_DIM + i] * sigma_u_host[i],
                    control_compute[j*num_timesteps*CONTROL_DIM + timestep * CONTROL_DIM + i]);
        }
    }

    // Make sure everything else are initial control plus noise
    for (int j = 1; j < num_rollouts*.99; ++j) {
        for (int i = 0; i < CONTROL_DIM; ++i) {
            ASSERT_FLOAT_EQ(ep_v_host[j*num_timesteps*CONTROL_DIM + timestep * CONTROL_DIM + i] * sigma_u_host[i] + u_traj_host[i],
                            control_compute[j*num_timesteps*CONTROL_DIM + timestep * CONTROL_DIM + i]) << "Failed at rollout number: " << j;
        }
    }
}

TEST(RolloutKernel, computeRunningCostAllRollouts) {
    // Instantiate the cost object.
    CartPoleQuadraticCost cost;
    cost.GPUSetup();

    const int num_timesteps = 100;
    const int num_rollouts = 100;
    const int state_dim = 4;
    const int control_dim = 1;
    float dt = 0.01;

    // Generate the trajectory data
    std::array<float, state_dim*num_timesteps*num_rollouts> x_traj = {0};
    x_traj.fill(2.f);
    std::array<float, control_dim*num_timesteps*num_rollouts> u_traj = {1.f};
    u_traj.fill(1.f);
    std::array<float, control_dim*num_timesteps*num_rollouts> du_traj = {0.9f};
    du_traj.fill(0.9f);
    std::array<float, control_dim> sigma_u = {0.1f};
    std::array<float, num_rollouts> cost_compute = {0.f};
    std::array<float, num_rollouts> cost_known = {0.f};

    // Compute the result on the CPU side
    computeRunningCostAllRollouts_CPU_TEST<CartPoleQuadraticCost, num_rollouts, num_timesteps, state_dim, control_dim>(cost, dt, x_traj, u_traj, du_traj, sigma_u, cost_known);

    // Launch the GPU kernel
    launchComputeRunningCostAllRollouts_KernelTest<CartPoleQuadraticCost, num_rollouts, num_timesteps, state_dim, control_dim>(cost, dt, x_traj, u_traj, du_traj, sigma_u, cost_compute);

    // Compare
    array_expect_float_eq<num_rollouts>(cost_known, cost_compute);

}

TEST(RolloutKernel, incrementStateAllRollouts) {
    FAIL();
}

