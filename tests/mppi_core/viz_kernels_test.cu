#include <gtest/gtest.h>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/core/mppi_common.cuh>

#include <mppi/utils/test_helper.h>
#include <random>

class VizualizationKernelsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
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

    cudaStreamCreate(&stream);

    /**
     * Ensure dynamics and costs exist on GPU
     */
    dynamics.bindToStream(stream);
    cost.bindToStream(stream);
    // Call the GPU setup functions of the model and cost
    dynamics.GPUSetup();
    cost.GPUSetup();
    fb_controller.GPUSetup();

    for (int i = 0; i < x0.rows(); i++)
    {
      x0(i) = i * 0.1 + 0.2;
    }

    // Create x init cuda array
    HANDLE_ERROR(cudaMalloc((void**)&initial_state_d, sizeof(float) * CartpoleDynamics::STATE_DIM * 2));
    // create control u trajectory cuda array
    HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float) * CartpoleDynamics::CONTROL_DIM * num_timesteps * 2));
    // Create cost trajectory cuda array
    HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d, sizeof(float) * num_rollouts * 2));
    // Create result state cuda array
    HANDLE_ERROR(cudaMalloc((void**)&result_state_d, sizeof(float) * num_rollouts * CartpoleDynamics::STATE_DIM * 2));
    // Create result state cuda array
    HANDLE_ERROR(cudaMalloc((void**)&crash_status_d, sizeof(float) * num_rollouts * 2));

    // Create random noise generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /**
     * Fill in GPU arrays
     */
    HANDLE_ERROR(cudaMemcpyAsync(initial_state_d, x0.data(), sizeof(float) * CartpoleDynamics::STATE_DIM,
                                 cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(initial_state_d + CartpoleDynamics::STATE_DIM, x0.data(),
                                 sizeof(float) * CartpoleDynamics::STATE_DIM, cudaMemcpyHostToDevice, stream));

    int control_noise_size = num_rollouts * num_timesteps * CartpoleDynamics::CONTROL_DIM;
    // Ensure copying finishes?
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }

  void TearDown() override
  {
    cudaFree(initial_state_d);
    cudaFree(control_d);
    cudaFree(trajectory_costs_d);
    cudaFree(crash_status_d);
    cudaFree(result_state_d);
  }

  CartpoleDynamics dynamics = CartpoleDynamics(1, 1, 1);
  CartpoleQuadraticCost cost;
  DDPFeedback<CartpoleDynamics, 10> fb_controller = DDPFeedback<CartpoleDynamics, 10>(&dynamics, dt);

  float dt = 0.2;
  int num_timesteps = 10;
  int num_rollouts = 2;
  float lambda = 0.5;
  float alpha = 0.001;

  cudaStream_t stream;
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y = 4;

  float* initial_state_d;
  float* control_d;
  float* trajectory_costs_d;
  float* result_state_d;
  int* crash_status_d;

  CartpoleDynamics::state_array x0;

  std::vector<float> initial_state = std::vector<float>(CartpoleDynamics::STATE_DIM);
  std::vector<float> control = std::vector<float>(num_rollouts * CartpoleDynamics::CONTROL_DIM);
  std::vector<float> result_state = std::vector<float>(num_rollouts * CartpoleDynamics::STATE_DIM);
  std::vector<float> trajectory_costs = std::vector<float>(num_rollouts);
  std::vector<int> crash_status = std::vector<int>(num_rollouts);
};

TEST_F(VizualizationKernelsTest, stateAndCostTrajectoryKernelNoZNoFeedbackTest)
{
  // Launch rollout kernel
  mppi_common::launchStateAndCostTrajectoryKernel<CartpoleDynamics, CartpoleQuadraticCost,
                                                  DeviceDDP<CartpoleDynamics, 10>, 32, 4>(
      dynamics.model_d_, cost.cost_d_, fb_controller.getDevicePointer(), control_d, initial_state_d, result_state_d,
      trajectory_costs_d, crash_status_d, num_rollouts, num_timesteps, dt, stream);

  // Copy the costs back to the host
  HANDLE_ERROR(cudaMemcpyAsync(result_state.data(), result_state_d,
                               num_rollouts * CartpoleDynamics::STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs.data(), trajectory_costs_d, num_rollouts * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(
      cudaMemcpyAsync(crash_status.data(), crash_status_d, num_rollouts * sizeof(int), cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));
}
