#include <gtest/gtest.h>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/core/rollout_kernel_test.cuh>
#include <mppi/utils/test_helper.h>
#include <random>
/*
 * Here we will test various device functions that are related to cuda kernel things.
 */

TEST(RolloutKernel, loadGlobalToShared) {
    const int STATE_DIM = 12;
    const int CONTROL_DIM = 3;
    std::vector<float> x0_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<float> u_var_host = {0.8, 0.9, 1.0};

    std::vector<float> x_thread_host(STATE_DIM, 0.f);
    std::vector<float> xdot_thread_host(STATE_DIM, 2.f);

    std::vector<float> u_thread_host(CONTROL_DIM, 3.f);
    std::vector<float> du_thread_host(CONTROL_DIM, 4.f);
    std::vector<float> sigma_u_thread_host(CONTROL_DIM, 0.f);

    launchGlobalToShared_KernelTest(x0_host, u_var_host, x_thread_host, xdot_thread_host, u_thread_host, du_thread_host, sigma_u_thread_host);

    array_assert_float_eq(x0_host, x_thread_host, STATE_DIM);
    array_assert_float_eq(0.f, xdot_thread_host, STATE_DIM);
    array_assert_float_eq(0.f, u_thread_host, CONTROL_DIM);
    array_assert_float_eq(0.f, du_thread_host, CONTROL_DIM);
    array_assert_float_eq(sigma_u_thread_host, u_var_host, CONTROL_DIM);
}

TEST(RolloutKernel, loadGlobalToSharedNominalAndActualState) {
  const int STATE_DIM = 12;
  const int CONTROL_DIM = 3;
  std::vector<float> x0_host_act = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                    1.0, 1.1, 1.2};

  std::vector<float> x0_host_nom = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                    2.0, 2.1, 2.2};
  std::vector<float> u_var_host = {0.8, 0.9, 1.0};

  std::vector<float> x_thread_host_act(STATE_DIM, 0.f);
  std::vector<float> x_thread_host_nom(STATE_DIM, 0.f);
  std::vector<float> xdot_thread_host_act(STATE_DIM, 2.f);
  std::vector<float> xdot_thread_host_nom(STATE_DIM, 2.f);

  std::vector<float> u_thread_host_act(CONTROL_DIM, 3.f);
  std::vector<float> u_thread_host_nom(CONTROL_DIM, 3.f);
  std::vector<float> du_thread_host_act(CONTROL_DIM, 4.f);
  std::vector<float> du_thread_host_nom(CONTROL_DIM, 4.f);
  std::vector<float> sigma_u_thread_host(CONTROL_DIM, 0.f);

  launchGlobalToShared_KernelTest_nom_act(x0_host_act, u_var_host,
                                          x_thread_host_act, xdot_thread_host_act,
                                          u_thread_host_act, du_thread_host_act,
                                          x0_host_nom, x_thread_host_nom,
                                          xdot_thread_host_nom,
                                          u_thread_host_nom, du_thread_host_nom,
                                          sigma_u_thread_host);

  //std::cout << "Testing actual x0" << std::endl;
  array_assert_float_eq(x0_host_act, x_thread_host_act, STATE_DIM);
  //std::cout << "Testing nom x0" << std::endl;
  array_assert_float_eq(x0_host_nom, x_thread_host_nom, STATE_DIM);
  //std::cout << "Testing empty" << std::endl;
  array_assert_float_eq(0.f, xdot_thread_host_act, STATE_DIM);
  array_assert_float_eq(0.f, xdot_thread_host_nom, STATE_DIM);
  array_assert_float_eq(0.f, u_thread_host_act, CONTROL_DIM);
  array_assert_float_eq(0.f, u_thread_host_nom, CONTROL_DIM);
  array_assert_float_eq(0.f, du_thread_host_act, CONTROL_DIM);
  array_assert_float_eq(0.f, du_thread_host_nom, CONTROL_DIM);
  //std::cout << "Testing act sigma" << std::endl;
  array_assert_float_eq(sigma_u_thread_host, u_var_host, CONTROL_DIM);
}

TEST(RolloutKernel, injectControlNoiseOnce) {
    const int NUM_ROLLOUTS = 1000;
    const int CONTROL_DIM = 3;
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

// TEST(RolloutKernel, injectControlNoiseAllTimeSteps) {
//     GTEST_SKIP() << "Not implemented";
// }

TEST(RolloutKernel, injectControlNoiseCheckControl_V) {
    const int num_rollouts = 100;
    const int control_dim = 3;
    const int num_timesteps = 5;
    std::array<float, control_dim*num_timesteps> u_traj_host = {0};
    // Control variance for each control channel
    std::array<float, control_dim> sigma_u_host = {0.1f, 0.2f, 0.3f};
    // Noise
    std::array<float, num_rollouts*num_timesteps*control_dim> ep_v_host = {0.f};
    std::array<float, num_rollouts*num_timesteps*control_dim> ep_v_compute = {0.f};

    auto generator = std::default_random_engine(7.0);
    auto distribution = std::normal_distribution<float>(5.0, 0.2);

    for (size_t i = 0; i <ep_v_host.size(); ++i) {
        ep_v_host[i] = distribution(generator);
        ep_v_compute[i] = ep_v_host[i];
    }

    for (size_t i = 0; i <u_traj_host.size(); ++i) {
        u_traj_host[i] = i;
    }

    // Output vector

    // CPU known vector
    std::array<float, num_rollouts*num_timesteps*control_dim> ep_v_known = {0.f};

    for (int i = 0; i < num_rollouts; ++i) {
        for (int j =0; j < num_timesteps; ++j) {
            for (int k =0; k < control_dim; ++k) {
                int index = i*num_timesteps*control_dim + j*control_dim + k;
                if (i == 0) {

                    ep_v_known[index] = u_traj_host[j*control_dim + k];
                } else if (i >= num_rollouts*.99) {
                    ep_v_known[index] = ep_v_host[index]*sigma_u_host[k];
                } else {
                    ep_v_known[index] = u_traj_host[j*control_dim + k] + ep_v_host[index]*sigma_u_host[k];
                }
            }
        }
    }

    launchInjectControlNoiseCheckControlV_KernelTest<num_rollouts,
    num_timesteps, control_dim, 64, 8, num_rollouts>(u_traj_host, ep_v_compute, sigma_u_host);

    array_assert_float_eq<num_rollouts*num_timesteps*control_dim>(ep_v_known, ep_v_compute);

}

TEST(RolloutKernel, computeAndSaveCostAllRollouts) {
    // Define an assortment of costs for a given number of rollouts
    CartpoleQuadraticCost cost;
    cost.GPUSetup();

    const int num_rollouts = 1234;
    std::array<float, num_rollouts> cost_all_rollouts = {0};
    std::array<float, CartpoleDynamics::STATE_DIM * num_rollouts> x_traj_terminal = {0};
    std::array<float, num_rollouts> cost_known = {0};
    std::array<float, num_rollouts> cost_compute = {0};

    std::default_random_engine generator(7.0);
    std::normal_distribution<float> distribution(1.0,2.0);

    for(auto& costs: cost_all_rollouts) {
        costs = 10*distribution(generator);
    }

    for (auto& state: x_traj_terminal) {
        state = distribution(generator);
    }
    // Compute terminal cost on CPU
    for (int i = 0; i < num_rollouts; ++i) {
        cost_known[i] = cost_all_rollouts[i] +
                (x_traj_terminal[CartpoleDynamics::STATE_DIM * i] * x_traj_terminal[CartpoleDynamics::STATE_DIM * i] * cost.getParams().cart_position_coeff +
                 x_traj_terminal[CartpoleDynamics::STATE_DIM * i + 1] * x_traj_terminal[CartpoleDynamics::STATE_DIM * i + 1] * cost.getParams().cart_velocity_coeff +
                 x_traj_terminal[CartpoleDynamics::STATE_DIM * i + 2] * x_traj_terminal[CartpoleDynamics::STATE_DIM * i + 2] * cost.getParams().pole_angle_coeff +
                 x_traj_terminal[CartpoleDynamics::STATE_DIM * i + 3] * x_traj_terminal[CartpoleDynamics::STATE_DIM * i + 3] * cost.getParams().pole_angular_velocity_coeff) *
                cost.getParams().terminal_cost_coeff;
    }

    // Compute the dynamics on the GPU
    launchComputeAndSaveCostAllRollouts_KernelTest<CartpoleQuadraticCost, CartpoleDynamics::STATE_DIM, num_rollouts>(cost,
                                                                                                                     cost_all_rollouts, x_traj_terminal, cost_compute);

    array_assert_float_eq<num_rollouts>(cost_known, cost_compute);
}

TEST(RolloutKernel, runRolloutKernelOnMultipleSystems) {
  CartpoleDynamics dynamics(1, 1, 1);
  CartpoleQuadraticCost cost;

  CartpoleQuadraticCostParams new_params;
  new_params.cart_position_coeff = 100;
  new_params.pole_angle_coeff = 200;
  new_params.cart_velocity_coeff = 10;
  new_params.pole_angular_velocity_coeff = 20;
  new_params.control_cost_coeff[0] = 1;
  new_params.terminal_cost_coeff = 0;
  new_params.desired_terminal_state[0] = -20;
  new_params.desired_terminal_state[1] = 0;
  new_params.desired_terminal_state[2] = M_PI;
  new_params.desired_terminal_state[3] = 0;

  cost.setParams(new_params);

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  float dt = 0.01;
  int num_timesteps = 100;
  const int NUM_ROLLOUTS = 2048; // Must be a multiple of 32
  // Create variables to pass to rolloutKernel
  std::vector<float> x0(CartpoleDynamics::STATE_DIM);
  std::vector<float> control_std_dev(CartpoleDynamics::CONTROL_DIM, 0.4);
  std::vector<float> nominal_control_seq(CartpoleDynamics::CONTROL_DIM * num_timesteps);
  std::vector<float> trajectory_costs_act(NUM_ROLLOUTS);
  std::vector<float> trajectory_costs_nom(NUM_ROLLOUTS);
  for (size_t i = 0; i < x0.size(); i++) {
    x0[i] = i * 0.1 + 0.2;
  }
  float lambda = 0.5;
  float alpha = 0.001;

  launchRolloutKernel_nom_act<CartpoleDynamics, CartpoleQuadraticCost, NUM_ROLLOUTS>(
          &dynamics, &cost, dt, num_timesteps, lambda, alpha, x0, control_std_dev,
          nominal_control_seq, trajectory_costs_act, trajectory_costs_nom);
  array_assert_float_eq(trajectory_costs_act, trajectory_costs_nom, NUM_ROLLOUTS);
}
