#include "rollout_kernel_test.cuh"

#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>

// const int STATE_DIM = 12;
// const int CONTROL_DIM = 3;
// const int NUM_ROLLOUTS = 100; // .99 times this number has to be an integer... TODO fix how brittle this is
// const int BLOCKSIZE_X = 32;
// const int BLOCKSIZE_Y = 8; // Blocksize_y has to be greater than the control dim TODO fix how we step through the
// controls

template <int BLOCKSIZE_Z>
__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device, float* x_thread_device,
                                              float* xdot_thread_device, float* u_thread_device,
                                              float* du_thread_device, float* sigma_u_thread_device)
{
  const int STATE_DIM = 12;
  const int CONTROL_DIM = 3;
  const int NUM_ROLLOUTS = 100;  // .99 times this number has to be an integer... TODO fix how brittle this is
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y =
      8;  // Blocksize_y has to be greater than the control dim TODO fix how we step through the controls

  int thread_idx = threadIdx.x;
  int thread_idy = threadIdx.y;
  int thread_idz = threadIdx.z;
  int block_idx = blockIdx.x;
  int global_idx = threadIdx.x + block_idx * blockDim.x;

  // Create shared arrays which hold state and control data
  __shared__ float x_shared[BLOCKSIZE_X * STATE_DIM * BLOCKSIZE_Z];
  __shared__ float xdot_shared[BLOCKSIZE_X * STATE_DIM * BLOCKSIZE_Z];
  __shared__ float u_shared[BLOCKSIZE_X * CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float du_shared[BLOCKSIZE_X * CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float sigma_u_thread[CONTROL_DIM];

  float* x_thread;
  float* xdot_thread;

  float* u_thread;
  float* du_thread;

  if (global_idx < NUM_ROLLOUTS)
  {
    x_thread = &x_shared[(blockDim.x * thread_idz + thread_idx) * STATE_DIM];
    xdot_thread = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * STATE_DIM];
    u_thread = &u_shared[(blockDim.x * thread_idz + thread_idx) * CONTROL_DIM];
    du_thread = &du_shared[(blockDim.x * thread_idz + thread_idx) * CONTROL_DIM];
  }
  __syncthreads();
  mppi_common::loadGlobalToShared<STATE_DIM, CONTROL_DIM>(NUM_ROLLOUTS, BLOCKSIZE_Y, global_idx, thread_idy, thread_idz,
                                                          x0_device, sigma_u_device, x_thread, xdot_thread, u_thread,
                                                          du_thread, sigma_u_thread);
  __syncthreads();

  // Check if on the first rollout the correct values were coped over
  // Prevent y threads from all writing to the same memory
  if (global_idx == 1 && thread_idy == 0)
  {
    for (int i = 0; i < STATE_DIM; ++i)
    {
      int ind = i + thread_idz * STATE_DIM;
      int ind_thread = i + thread_idz * STATE_DIM * blockDim.x;
      x_thread_device[ind] = x_shared[ind_thread];
      xdot_thread_device[ind] = xdot_shared[ind_thread];
    }

    for (int i = 0; i < CONTROL_DIM; ++i)
    {
      int ind = i + thread_idz * CONTROL_DIM;
      int ind_thread = i + thread_idz * CONTROL_DIM * blockDim.x;
      u_thread_device[ind] = u_shared[ind_thread];
      du_thread_device[ind] = du_shared[ind_thread];
      // There is only control_dim
      sigma_u_thread_device[i] = sigma_u_thread[i];
    }
    __syncthreads();
  }

  // To test what the results are, we have to return them back to the host.
}

void launchGlobalToShared_KernelTest(const std::vector<float>& x0_host, const std::vector<float>& u_var_host,
                                     std::vector<float>& x_thread_host, std::vector<float>& xdot_thread_host,
                                     std::vector<float>& u_thread_host, std::vector<float>& du_thread_host,
                                     std::vector<float>& sigma_u_thread_host)
{
  const int STATE_DIM = 12;
  const int CONTROL_DIM = 3;
  const int NUM_ROLLOUTS = 100;  // .99 times this number has to be an integer... TODO fix how brittle this is
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y =
      8;  // Blocksize_y has to be greater than the control dim TODO fix how we step through the controls

  // Define the initial condition x0_device and the exploration variance in global device memory
  float* x0_device;
  float* u_var_device;
  HANDLE_ERROR(cudaMalloc((void**)&x0_device, sizeof(float) * STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&u_var_device, sizeof(float) * CONTROL_DIM));

  HANDLE_ERROR(cudaMemcpy(x0_device, x0_host.data(), sizeof(float) * STATE_DIM, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(u_var_device, u_var_host.data(), sizeof(float) * CONTROL_DIM, cudaMemcpyHostToDevice));

  // Define the return arguments in global device memory
  float* x_thread_device;
  float* xdot_thread_device;
  float* u_thread_device;
  float* du_thread_device;
  float* sigma_u_thread_device;

  HANDLE_ERROR(cudaMalloc((void**)&x_thread_device, sizeof(float) * STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&xdot_thread_device, sizeof(float) * STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&u_thread_device, sizeof(float) * CONTROL_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&du_thread_device, sizeof(float) * CONTROL_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&sigma_u_thread_device, sizeof(float) * CONTROL_DIM));

  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
  dim3 dimGrid(2048);

  loadGlobalToShared_KernelTest<<<dimGrid, dimBlock>>>(x0_device, u_var_device, x_thread_device, xdot_thread_device,
                                                       u_thread_device, du_thread_device, sigma_u_thread_device);
  CudaCheckError();

  // Copy the data back to the host
  HANDLE_ERROR(cudaMemcpy(x_thread_host.data(), x_thread_device, sizeof(float) * STATE_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(xdot_thread_host.data(), xdot_thread_device, sizeof(float) * STATE_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(u_thread_host.data(), u_thread_device, sizeof(float) * CONTROL_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(du_thread_host.data(), du_thread_device, sizeof(float) * CONTROL_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(sigma_u_thread_host.data(), sigma_u_thread_device, sizeof(float) * CONTROL_DIM,
                          cudaMemcpyDeviceToHost));

  // Free the cuda memory that we allocated
  cudaFree(x0_device);
  cudaFree(u_var_device);

  cudaFree(x_thread_device);
  cudaFree(xdot_thread_device);
  cudaFree(u_thread_device);
  cudaFree(du_thread_device);
  cudaFree(sigma_u_thread_device);
}

/**
 * Tube-MPPI versions of the kernel tests
 */

// This is to test tube-mppi calls to the kernel
void launchGlobalToShared_KernelTest_nom_act(
    const std::vector<float>& x0_host_act, const std::vector<float>& u_var_host, std::vector<float>& x_thread_host_act,
    std::vector<float>& xdot_thread_host_act, std::vector<float>& u_thread_host_act,
    std::vector<float>& du_thread_host_act, const std::vector<float>& x0_host_nom,
    std::vector<float>& x_thread_host_nom, std::vector<float>& xdot_thread_host_nom,
    std::vector<float>& u_thread_host_nom, std::vector<float>& du_thread_host_nom,
    std::vector<float>& sigma_u_thread_host)
{
  const int STATE_DIM = 12;
  const int CONTROL_DIM = 3;
  const int NUM_ROLLOUTS = 100;  // .99 times this number has to be an integer... TODO fix how brittle this is
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y =
      8;  // Blocksize_y has to be greater than the control dim TODO fix how we step through the controls

  // Define the initial condition x0_device and the exploration variance in global device memory
  // Need twice as much memory for tube-mppi
  float* x0_device;
  float* u_var_device;
  HANDLE_ERROR(cudaMalloc((void**)&x0_device, sizeof(float) * STATE_DIM * 2));
  HANDLE_ERROR(cudaMalloc((void**)&u_var_device, sizeof(float) * CONTROL_DIM));

  // Copy both act and nominal initial state
  HANDLE_ERROR(cudaMemcpy(x0_device, x0_host_act.data(), sizeof(float) * STATE_DIM, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(x0_device + STATE_DIM, x0_host_nom.data(), sizeof(float) * STATE_DIM, cudaMemcpyHostToDevice));
  // Copy both act and nominal control variance
  HANDLE_ERROR(cudaMemcpy(u_var_device, u_var_host.data(), sizeof(float) * CONTROL_DIM, cudaMemcpyHostToDevice));

  // Define the return arguments in global device memory
  float* x_thread_device;
  float* xdot_thread_device;
  float* u_thread_device;
  float* du_thread_device;
  float* sigma_u_thread_device;

  HANDLE_ERROR(cudaMalloc((void**)&x_thread_device, sizeof(float) * STATE_DIM * 2));
  HANDLE_ERROR(cudaMalloc((void**)&xdot_thread_device, sizeof(float) * STATE_DIM * 2));
  HANDLE_ERROR(cudaMalloc((void**)&u_thread_device, sizeof(float) * CONTROL_DIM * 2));
  HANDLE_ERROR(cudaMalloc((void**)&du_thread_device, sizeof(float) * CONTROL_DIM * 2));
  HANDLE_ERROR(cudaMalloc((void**)&sigma_u_thread_device, sizeof(float) * CONTROL_DIM));

  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 2);
  dim3 dimGrid(100);

  loadGlobalToShared_KernelTest<2><<<dimGrid, dimBlock>>>(x0_device, u_var_device, x_thread_device, xdot_thread_device,
                                                          u_thread_device, du_thread_device, sigma_u_thread_device);
  CudaCheckError();

  // Copy the initial_state for actual and nominal
  HANDLE_ERROR(
      cudaMemcpy(x_thread_host_act.data(), x_thread_device, sizeof(float) * STATE_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(x_thread_host_nom.data(), x_thread_device + STATE_DIM, sizeof(float) * STATE_DIM,
                          cudaMemcpyDeviceToHost));
  // Copy the xdot for actual and nominal
  HANDLE_ERROR(
      cudaMemcpy(xdot_thread_host_act.data(), xdot_thread_device, sizeof(float) * STATE_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(xdot_thread_host_nom.data(), xdot_thread_device + STATE_DIM, sizeof(float) * STATE_DIM,
                          cudaMemcpyDeviceToHost));
  // copy the initial u for actual and nominal
  HANDLE_ERROR(
      cudaMemcpy(u_thread_host_act.data(), u_thread_device, sizeof(float) * CONTROL_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(u_thread_host_nom.data(), u_thread_device + CONTROL_DIM, sizeof(float) * CONTROL_DIM,
                          cudaMemcpyDeviceToHost));
  // copy the du for actual and nominal
  HANDLE_ERROR(
      cudaMemcpy(du_thread_host_act.data(), du_thread_device, sizeof(float) * CONTROL_DIM, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(du_thread_host_nom.data(), du_thread_device + CONTROL_DIM, sizeof(float) * CONTROL_DIM,
                          cudaMemcpyDeviceToHost));
  // copy the control variance for actual and nominal
  HANDLE_ERROR(cudaMemcpy(sigma_u_thread_host.data(), sigma_u_thread_device, sizeof(float) * CONTROL_DIM,
                          cudaMemcpyDeviceToHost));

  // Free the cuda memory that we allocated
  cudaFree(x0_device);
  cudaFree(u_var_device);

  cudaFree(x_thread_device);
  cudaFree(xdot_thread_device);
  cudaFree(u_thread_device);
  cudaFree(du_thread_device);
  cudaFree(sigma_u_thread_device);
}

__global__ void injectControlNoiseOnce_KernelTest(int num_rollouts, int num_timesteps, int timestep,
                                                  float* u_traj_device, float* ep_v_device, float* sigma_u_device,
                                                  float* control_compute_device)
{
  const int STATE_DIM = 12;
  const int CONTROL_DIM = 3;
  const int NUM_ROLLOUTS = 100;  // .99 times this number has to be an integer... TODO fix how brittle this is
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y =
      8;  // Blocksize_y has to be greater than the control dim TODO fix how we step through the controls

  int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int thread_idy = threadIdx.y;
  float u_thread[CONTROL_DIM];
  float du_thread[CONTROL_DIM];

  if (global_idx < num_rollouts)
  {
    mppi_common::injectControlNoise(CONTROL_DIM, BLOCKSIZE_Y, num_rollouts, num_timesteps, timestep, global_idx,
                                    thread_idy, 0, u_traj_device, ep_v_device, sigma_u_device, u_thread, du_thread);
    if (thread_idy < CONTROL_DIM)
    {
      control_compute_device[global_idx * CONTROL_DIM + thread_idy] = u_thread[thread_idy];
    }
  }
}

void launchInjectControlNoiseOnce_KernelTest(const std::vector<float>& u_traj_host, const int num_rollouts,
                                             const int num_timesteps, std::vector<float>& ep_v_host,
                                             std::vector<float>& sigma_u_host, std::vector<float>& control_compute)
{
  const int STATE_DIM = 12;
  const int CONTROL_DIM = 3;
  const int NUM_ROLLOUTS = 100;  // .99 times this number has to be an integer... TODO fix how brittle this is
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y =
      8;  // Blocksize_y has to be greater than the control dim TODO fix how we step through the controls
  // Timestep
  int timestep = 0;

  // Declare variables for device memory
  float* u_traj_device;
  float* ep_v_device;
  float* sigma_u_device;
  float* control_compute_device;

  // Allocate cuda memory
  HANDLE_ERROR(cudaMalloc((void**)&u_traj_device, sizeof(float) * u_traj_host.size()));
  HANDLE_ERROR(cudaMalloc((void**)&ep_v_device, sizeof(float) * ep_v_host.size()));
  HANDLE_ERROR(cudaMalloc((void**)&sigma_u_device, sizeof(float) * sigma_u_host.size()));
  HANDLE_ERROR(cudaMalloc((void**)&control_compute_device, sizeof(float) * control_compute.size()));

  // Copy the control trajectory and the control variance to the device
  HANDLE_ERROR(
      cudaMemcpy(u_traj_device, u_traj_host.data(), sizeof(float) * u_traj_host.size(), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(sigma_u_device, sigma_u_host.data(), sizeof(float) * sigma_u_host.size(), cudaMemcpyHostToDevice));

  // Generate the noise
  curandGenerator_t gen_;
  curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL);
  curandGenerateNormal(gen_, ep_v_device, ep_v_host.size(), 0.0, 1.0);

  // Copy the noise back to the host
  HANDLE_ERROR(cudaMemcpy(ep_v_host.data(), ep_v_device, sizeof(float) * ep_v_host.size(), cudaMemcpyDeviceToHost));

  // Create the block and grid dimensions
  dim3 block_size(BLOCKSIZE_X, BLOCKSIZE_Y);
  dim3 grid_size(num_rollouts, 1);

  // Launch the test kernel
  injectControlNoiseOnce_KernelTest<<<grid_size, block_size>>>(num_rollouts, num_timesteps, timestep, u_traj_device,
                                                               ep_v_device, sigma_u_device, control_compute_device);
  CudaCheckError();

  // Copy the result back to the host
  HANDLE_ERROR(cudaMemcpy(control_compute.data(), control_compute_device, sizeof(float) * control_compute.size(),
                          cudaMemcpyDeviceToHost));

  // Free cuda memory
  cudaFree(u_traj_device);
  cudaFree(ep_v_device);
  cudaFree(control_compute_device);
  cudaFree(sigma_u_device);
  curandDestroyGenerator(gen_);
}

template <int control_dim, int blocksize_y>
__global__ void injectControlNoiseCheckControlV_KernelTest(int num_rollouts, int num_timesteps, int timestep,
                                                           float* u_traj_device, float* ep_v_device,
                                                           float* sigma_u_device)
{
  int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int thread_idy = threadIdx.y;
  float u_thread[control_dim];
  float du_thread[control_dim];

  if (global_idx < num_rollouts)
  {
    mppi_common::injectControlNoise(control_dim, blocksize_y, num_rollouts, num_timesteps, timestep, global_idx,
                                    thread_idy, 1, u_traj_device, ep_v_device, sigma_u_device, u_thread, du_thread);
    __syncthreads();
  }
}

template <int num_rollouts, int num_timesteps, int control_dim, int blocksize_x, int blocksize_y, int gridsize_x>
void launchInjectControlNoiseCheckControlV_KernelTest(
    const std::array<float, num_timesteps * control_dim>& u_traj_host,
    std::array<float, num_rollouts * num_timesteps * control_dim>& ep_v_host,
    const std::array<float, control_dim>& sigma_u_host)
{
  // Declare variables for device memory
  float* u_traj_device;
  float* ep_v_device;
  float* sigma_u_device;

  // Allocate cuda memory
  HANDLE_ERROR(cudaMalloc((void**)&u_traj_device, sizeof(float) * u_traj_host.size()));
  HANDLE_ERROR(cudaMalloc((void**)&ep_v_device, sizeof(float) * ep_v_host.size()));
  HANDLE_ERROR(cudaMalloc((void**)&sigma_u_device, sizeof(float) * sigma_u_host.size()));

  // Copy to device
  HANDLE_ERROR(
      cudaMemcpy(u_traj_device, u_traj_host.data(), sizeof(float) * u_traj_host.size(), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(ep_v_device, ep_v_host.data(), sizeof(float) * ep_v_host.size(), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(sigma_u_device, sigma_u_host.data(), sizeof(float) * sigma_u_host.size(), cudaMemcpyHostToDevice));

  dim3 dimBlock(blocksize_x, blocksize_y, 1);
  dim3 dimGrid(gridsize_x, 1, 1);

  for (int i = 0; i < num_timesteps; ++i)
  {
    injectControlNoiseCheckControlV_KernelTest<control_dim, blocksize_y>
        <<<dimGrid, dimBlock>>>(num_rollouts, num_timesteps, i, u_traj_device, ep_v_device, sigma_u_device);
    CudaCheckError();
  }

  // Copy to the host
  HANDLE_ERROR(cudaMemcpy(ep_v_host.data(), ep_v_device, sizeof(float) * ep_v_host.size(), cudaMemcpyDeviceToHost));

  cudaFree(u_traj_device);
  cudaFree(ep_v_device);
  cudaFree(sigma_u_device);
}

template <class COST_T>
__global__ void computeAndSaveCostAllRollouts_KernelTest(COST_T* cost, int state_dim, int num_rollouts,
                                                         float* running_costs, float* terminal_state,
                                                         float* cost_rollout_device)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;  // index on rollouts
                                                    //    if (tid == 0) {
  //        printf("Current state [%f, %f, %f, %f]\n", terminal_state[state_dim * tid],
  //               terminal_state[state_dim * tid + 1], terminal_state[state_dim * tid + 2],
  //               terminal_state[state_dim * tid + 3]);
  //        printf("Current cost [%f]\n", running_costs[tid]);
  //    }
  mppi_common::computeAndSaveCost(num_rollouts, 2, tid, cost, &terminal_state[state_dim * tid], running_costs[tid],
                                  nullptr, cost_rollout_device);
  //    if (tid == 0) {
  //        printf("Total cost [%f]\n", cost_rollout_device[tid]);
  //    }
}

template <class COST_T, int STATE_DIM, int NUM_ROLLOUTS>
void launchComputeAndSaveCostAllRollouts_KernelTest(COST_T& cost,
                                                    const std::array<float, NUM_ROLLOUTS>& cost_all_rollouts,
                                                    const std::array<float, STATE_DIM * NUM_ROLLOUTS>& terminal_states,
                                                    std::array<float, NUM_ROLLOUTS>& cost_compute)
{
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y = 8;

  // Allocate CUDA memory
  float* cost_all_rollouts_device;
  float* terminal_states_device;
  float* cost_compute_device;

  HANDLE_ERROR(cudaMalloc((void**)&cost_all_rollouts_device, sizeof(float) * cost_all_rollouts.size()));
  HANDLE_ERROR(cudaMalloc((void**)&terminal_states_device, sizeof(float) * terminal_states.size()));
  HANDLE_ERROR(cudaMalloc((void**)&cost_compute_device, sizeof(float) * cost_compute.size()));

  // Copy Host side data to the Device
  HANDLE_ERROR(cudaMemcpy(cost_all_rollouts_device, cost_all_rollouts.data(), sizeof(float) * cost_all_rollouts.size(),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(terminal_states_device, terminal_states.data(), sizeof(float) * terminal_states.size(),
                          cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 blocksize(BLOCKSIZE_X, 1);
  dim3 gridsize((NUM_ROLLOUTS + (BLOCKSIZE_X - 1)) / BLOCKSIZE_X, 1);
  computeAndSaveCostAllRollouts_KernelTest<<<blocksize, gridsize>>>(
      cost.cost_d_, STATE_DIM, NUM_ROLLOUTS, cost_all_rollouts_device, terminal_states_device, cost_compute_device);
  CudaCheckError();

  // Copy Device side data to the host
  HANDLE_ERROR(cudaMemcpy(cost_compute.data(), cost_compute_device, sizeof(float) * cost_compute.size(),
                          cudaMemcpyDeviceToHost));

  // free cuda Memory
  cudaFree(cost_all_rollouts_device);
  cudaFree(terminal_states_device);
  cudaFree(cost_compute_device);
}

template <class DYN_T, class COST_T, int NUM_ROLLOUTS>
void launchRolloutKernel_nom_act(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, float lambda, float alpha,
                                 const std::vector<float>& x0, const std::vector<float>& sigma_u,
                                 const std::vector<float>& nom_control_seq, std::vector<float>& trajectory_costs_act,
                                 std::vector<float>& trajectory_costs_nom, cudaStream_t stream)
{
  float* initial_state_d;
  float* trajectory_costs_d;
  float* control_noise_d;  // du
  float* control_std_dev_d;
  float* control_d;

  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y = 8;

  /**
   * Ensure dynamics and costs exist on GPU
   */
  dynamics->bindToStream(stream);
  costs->bindToStream(stream);
  // Call the GPU setup functions of the model and cost
  dynamics->GPUSetup();
  costs->GPUSetup();

  int control_noise_size = NUM_ROLLOUTS * num_timesteps * DYN_T::CONTROL_DIM;
  // Create x init cuda array
  HANDLE_ERROR(cudaMalloc((void**)&initial_state_d, sizeof(float) * DYN_T::STATE_DIM * 2));
  // Create control variance cuda array
  HANDLE_ERROR(cudaMalloc((void**)&control_std_dev_d, sizeof(float) * DYN_T::CONTROL_DIM));
  // create control u trajectory cuda array
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps * 2));
  // Create cost trajectory cuda array
  HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d, sizeof(float) * NUM_ROLLOUTS * 2));
  // Create zero-mean noise cuda array
  HANDLE_ERROR(
      cudaMalloc((void**)&control_noise_d, sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps * NUM_ROLLOUTS * 2));
  // Create random noise generator
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  /**
   * Fill in GPU arrays
   */
  HANDLE_ERROR(
      cudaMemcpyAsync(initial_state_d, x0.data(), sizeof(float) * DYN_T::STATE_DIM, cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_d + DYN_T::STATE_DIM, x0.data(), sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_std_dev_d, sigma_u.data(), sizeof(float) * DYN_T::CONTROL_DIM,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_d, nom_control_seq.data(), sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(control_d + num_timesteps * DYN_T::CONTROL_DIM, nom_control_seq.data(),
                               sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps, cudaMemcpyHostToDevice, stream));

  curandGenerateNormal(gen, control_noise_d, control_noise_size, 0.0, 1.0);
  HANDLE_ERROR(cudaMemcpyAsync(control_noise_d + control_noise_size, control_noise_d,
                               control_noise_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  // Ensure copying finishes?
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  // Launch rollout kernel
  mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BLOCKSIZE_X, BLOCKSIZE_Y, 2>(
      dynamics->model_d_, costs->cost_d_, dt, num_timesteps, 0, lambda, alpha, initial_state_d, control_d,
      control_noise_d, control_std_dev_d, trajectory_costs_d, stream);

  // Copy the costs back to the host
  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_act.data(), trajectory_costs_d, NUM_ROLLOUTS * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nom.data(), trajectory_costs_d + NUM_ROLLOUTS,
                               NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  cudaFree(initial_state_d);
  cudaFree(control_std_dev_d);
  cudaFree(control_d);
  cudaFree(trajectory_costs_d);
  cudaFree(control_noise_d);
}

template <class DYNAMICS_T, class COSTS_T, int NUM_ROLLOUTS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
__global__ void autorallyRolloutKernel(int num_timesteps, float* state_d, float* U_d, float* du_d, float* nu_d,
                                       float* costs_d, DYNAMICS_T* dynamics_model, COSTS_T* mppi_costs, int opt_delay,
                                       float lambda, float alpha, float dt)
{
  int i, j;
  int tdx = threadIdx.x;
  int tdy = threadIdx.y;
  int bdx = blockIdx.x;

  // Initialize the local state, controls, and noise
  float* s;
  float* s_der;
  float* u;
  float* nu;
  float* du;
  int* crash;

  const int STATE_DIM = DYNAMICS_T::STATE_DIM;
  const int CONTROL_DIM = DYNAMICS_T::CONTROL_DIM;
  const int SHARED_MEM_REQUEST_GRD = DYNAMICS_T::SHARED_MEM_REQUEST_GRD;
  const int SHARED_MEM_REQUEST_BLK = DYNAMICS_T::SHARED_MEM_REQUEST_BLK;

  // Create shared arrays for holding state and control data.
  __shared__ float state_shared[BLOCKSIZE_X * STATE_DIM];
  __shared__ float state_der_shared[BLOCKSIZE_X * STATE_DIM];
  __shared__ float control_shared[BLOCKSIZE_X * CONTROL_DIM];
  __shared__ float control_var_shared[BLOCKSIZE_X * CONTROL_DIM];
  __shared__ float exploration_variance[BLOCKSIZE_X * CONTROL_DIM];
  __shared__ int crash_status[BLOCKSIZE_X];
  // Create a shared array for the dynamics model to use
  __shared__ float theta[SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 + SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X];
  __shared__ float theta_c[COSTS_T::SHARED_MEM_REQUEST_GRD + COSTS_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X];
  __shared__ float y[DYNAMICS_T::OUTPUT_DIM];

  // Initialize trajectory cost
  float running_cost = 0;

  int global_idx = BLOCKSIZE_X * bdx + tdx;
  if (global_idx < NUM_ROLLOUTS)
  {
    // Portion of the shared array belonging to each x-thread index.
    s = &state_shared[tdx * STATE_DIM];
    s_der = &state_der_shared[tdx * STATE_DIM];
    u = &control_shared[tdx * CONTROL_DIM];
    du = &control_var_shared[tdx * CONTROL_DIM];
    nu = &exploration_variance[tdx * CONTROL_DIM];
    crash = &crash_status[tdx];
    // Load the initial state, nu, and zero the noise
    for (i = tdy; i < STATE_DIM; i += blockDim.y)
    {
      s[i] = state_d[i];
      s_der[i] = 0;
    }
    // Load nu
    for (i = tdy; i < CONTROL_DIM; i += blockDim.y)
    {
      u[i] = 0;
      du[i] = 0;
      nu[i] = nu_d[i];
    }
    crash[0] = 0;
  }
  __syncthreads();
  /*<----Start of simulation loop-----> */
  dynamics_model->initializeDynamics(s, u, y, theta, 0.0, dt);
  mppi_costs->initializeCosts(s, u, theta_c, 0.0, dt);
  for (i = 0; i < num_timesteps; i++)
  {
    if (global_idx < NUM_ROLLOUTS)
    {
      for (j = tdy; j < CONTROL_DIM; j += blockDim.y)
      {
        // Noise free rollout
        if (global_idx == 0 || i < opt_delay)
        {  // Don't optimize variables that are already being executed
          du[j] = 0.0;
          u[j] = U_d[i * CONTROL_DIM + j];
        }
        else if (global_idx >= .99 * NUM_ROLLOUTS)
        {
          du[j] = du_d[CONTROL_DIM * num_timesteps * (BLOCKSIZE_X * bdx + tdx) + i * CONTROL_DIM + j] * nu[j];
          u[j] = du[j];
        }
        else
        {
          du[j] = du_d[CONTROL_DIM * num_timesteps * (BLOCKSIZE_X * bdx + tdx) + i * CONTROL_DIM + j] * nu[j];
          u[j] = U_d[i * CONTROL_DIM + j] + du[j];
        }
        du_d[CONTROL_DIM * num_timesteps * (BLOCKSIZE_X * bdx + tdx) + i * CONTROL_DIM + j] = u[j];
      }
    }
    __syncthreads();
    dynamics_model->enforceConstraints(s, &du_d[(global_idx * num_timesteps + i) * CONTROL_DIM]);
    if (tdy == 0 && global_idx < NUM_ROLLOUTS)
    {
      dynamics_model->enforceConstraints(s, u);
    }
    __syncthreads();
    // Compute the cost of the being in the current state
    if (tdy == 0 && global_idx < NUM_ROLLOUTS && i > 0 && crash[0] > -1)
    {
      // Running average formula
      running_cost +=
          (mppi_costs->computeRunningCost(s, u, du, nu, lambda, alpha, i, theta_c, crash) - running_cost) / (1.0 * i);
      //      printf("AutoRa Current State rollout %i, timestep: %i: [%f, %f, %f, %f]\n", global_idx, i, s[0], s[1],
      //      s[2], s[3]); printf("AutoRa Running Cost rollout %i, timestep: %i: %f\n", global_idx, i, running_cost);
    }
    // Compute the dynamics
    if (global_idx < NUM_ROLLOUTS)
    {
      dynamics_model->computeStateDeriv(s, u, s_der, theta);
    }
    __syncthreads();
    // Update the state
    if (global_idx < NUM_ROLLOUTS)
    {
      dynamics_model->updateState(s, s_der, dt);
    }
    //    //Check to see if the rollout will result in a (physical) crash.
    //    if (tdy == 0 && global_idx < NUM_ROLLOUTS) {
    //      mppi_costs.getCrash(s, crash);
    //    }
  }
  /* <------- End of the simulation loop ----------> */
  if (global_idx < NUM_ROLLOUTS && tdy == 0)
  {  // Write cost results back to global memory.
    costs_d[(BLOCKSIZE_X)*bdx + tdx] = running_cost + mppi_costs->terminalCost(s, theta_c);
  }
}

template <class DYNAMICS_T, class COSTS_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
void launchAutorallyRolloutKernelTest(
    DYNAMICS_T* dynamics, COSTS_T* costs, float dt, float lambda, float alpha,
    std::array<float, DYNAMICS_T::STATE_DIM> state_array,
    std::array<float, NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM> control_array,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM> control_noise_array,
    std::array<float, DYNAMICS_T::CONTROL_DIM> sigma_u, std::array<float, NUM_ROLLOUTS>& costs_out,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM>& control_noise_out, int opt_delay,
    cudaStream_t stream)
{
  float* state_d;
  float* U_d;
  float* du_d;
  float* nu_d;
  float* costs_d;

  // Allocate CUDA memory for the rollout
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * state_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&U_d, sizeof(float) * control_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&du_d, sizeof(float) * control_noise_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&nu_d, sizeof(float) * sigma_u.size()));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * costs_out.size()));

  // Copy the initial values
  HANDLE_ERROR(
      cudaMemcpyAsync(state_d, state_array.data(), sizeof(float) * state_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(
      cudaMemcpyAsync(U_d, control_array.data(), sizeof(float) * control_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(du_d, control_noise_array.data(), sizeof(float) * control_noise_array.size(),
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(nu_d, sigma_u.data(), sizeof(float) * sigma_u.size(), cudaMemcpyHostToDevice, stream));

  const int GRIDSIZE_X = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;

  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  dim3 dimGrid(GRIDSIZE_X, 1, 1);

  autorallyRolloutKernel<DYNAMICS_T, COSTS_T, NUM_ROLLOUTS, BLOCKSIZE_X, BLOCKSIZE_Y>
      <<<dimGrid, dimBlock, 0, stream>>>(NUM_TIMESTEPS, state_d, U_d, du_d, nu_d, costs_d, dynamics->model_d_,
                                         costs->cost_d_, opt_delay, lambda, alpha, dt);

  CudaCheckError();

  // Copy data back
  HANDLE_ERROR(
      cudaMemcpyAsync(costs_out.data(), costs_d, sizeof(float) * costs_out.size(), cudaMemcpyDeviceToHost, stream));

  // Copy the noise back
  HANDLE_ERROR(cudaMemcpyAsync(control_noise_out.data(), du_d, sizeof(float) * control_noise_out.size(),
                               cudaMemcpyDeviceToHost, stream));

  // Deallocate CUDA Memory
  HANDLE_ERROR(cudaFree(state_d));
  HANDLE_ERROR(cudaFree(U_d));
  HANDLE_ERROR(cudaFree(du_d));
  HANDLE_ERROR(cudaFree(nu_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

template <class DYNAMICS_T, class COSTS_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
void launchGenericRolloutKernelTest(
    DYNAMICS_T* dynamics, COSTS_T* costs, float dt, float lambda, float alpha,
    std::array<float, DYNAMICS_T::STATE_DIM> state_array,
    std::array<float, NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM> control_array,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM> control_noise_array,
    std::array<float, DYNAMICS_T::CONTROL_DIM> sigma_u, std::array<float, NUM_ROLLOUTS>& costs_out,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM>& control_noise_out, int opt_delay,
    cudaStream_t stream)
{
  float* state_d;
  float* U_d;
  float* du_d;
  float* nu_d;
  float* costs_d;

  // Allocate CUDA memory for the rollout
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * state_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&U_d, sizeof(float) * control_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&du_d, sizeof(float) * control_noise_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&nu_d, sizeof(float) * sigma_u.size()));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * costs_out.size()));

  // Copy the initial values
  HANDLE_ERROR(
      cudaMemcpyAsync(state_d, state_array.data(), sizeof(float) * state_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(
      cudaMemcpyAsync(U_d, control_array.data(), sizeof(float) * control_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(du_d, control_noise_array.data(), sizeof(float) * control_noise_array.size(),
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(nu_d, sigma_u.data(), sizeof(float) * sigma_u.size(), cudaMemcpyHostToDevice, stream));

  const int gridsize_x = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  dim3 dimGrid(gridsize_x, 1, 1);
  mppi_common::rolloutKernel<DYNAMICS_T, COSTS_T, BLOCKSIZE_X, BLOCKSIZE_Y, NUM_ROLLOUTS, 1>
      <<<dimGrid, dimBlock, 0, stream>>>(dynamics->model_d_, costs->cost_d_, dt, NUM_TIMESTEPS, opt_delay, lambda,
                                         alpha, state_d, U_d, du_d, nu_d, costs_d);
  CudaCheckError();

  // Copy data back
  HANDLE_ERROR(
      cudaMemcpyAsync(costs_out.data(), costs_d, sizeof(float) * costs_out.size(), cudaMemcpyDeviceToHost, stream));

  // Copy the noise back
  HANDLE_ERROR(cudaMemcpyAsync(control_noise_out.data(), du_d, sizeof(float) * control_noise_out.size(),
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  // Deallocate CUDA Memory
  HANDLE_ERROR(cudaFree(state_d));
  HANDLE_ERROR(cudaFree(U_d));
  HANDLE_ERROR(cudaFree(du_d));
  HANDLE_ERROR(cudaFree(nu_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

template <class DYNAMICS_T, class COSTS_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
void launchFastRolloutKernelTest(
    DYNAMICS_T* dynamics, COSTS_T* costs, float dt, float lambda, float alpha,
    std::array<float, DYNAMICS_T::STATE_DIM>& state_array,
    std::array<float, NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM>& control_array,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM>& control_noise_array,
    std::array<float, DYNAMICS_T::CONTROL_DIM> sigma_u, std::array<float, NUM_ROLLOUTS>& costs_out,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM>& control_noise_out, int opt_delay,
    int state_traj_array_size, cudaStream_t stream)
{
  float* state_d;
  float* U_d;
  float* du_d;
  float* nu_d;
  float* costs_d;
  float* x_d;

  // Allocate CUDA memory for the rollout
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * state_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&U_d, sizeof(float) * control_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&du_d, sizeof(float) * control_noise_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&nu_d, sizeof(float) * sigma_u.size()));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * costs_out.size()));
  HANDLE_ERROR(cudaMalloc((void**)&x_d, sizeof(float) * state_traj_array_size));

  // Copy the initial values
  HANDLE_ERROR(
      cudaMemcpyAsync(state_d, state_array.data(), sizeof(float) * state_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(
      cudaMemcpyAsync(U_d, control_array.data(), sizeof(float) * control_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(du_d, control_noise_array.data(), sizeof(float) * control_noise_array.size(),
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(nu_d, sigma_u.data(), sizeof(float) * sigma_u.size(), cudaMemcpyHostToDevice, stream));

  mppi_common::launchFastRolloutKernel<DYNAMICS_T, COSTS_T, NUM_ROLLOUTS, BLOCKSIZE_X, BLOCKSIZE_Y>(
      dynamics->model_d_, costs->cost_d_, dt, NUM_TIMESTEPS, opt_delay, lambda, alpha, state_d, x_d, U_d, du_d, nu_d,
      costs_d, stream, true);

  // Copy data back
  HANDLE_ERROR(
      cudaMemcpyAsync(costs_out.data(), costs_d, sizeof(float) * costs_out.size(), cudaMemcpyDeviceToHost, stream));

  // Copy the noise back
  HANDLE_ERROR(cudaMemcpyAsync(control_noise_out.data(), du_d, sizeof(float) * control_noise_out.size(),
                               cudaMemcpyDeviceToHost, stream));
  // HANDLE_ERROR(cudaMemcpyAsync(state_traj_array.data(), x_d, sizeof(float) * state_traj_array.size(),
  //                              cudaMemcpyDeviceToHost, stream));

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  // Deallocate CUDA Memory
  HANDLE_ERROR(cudaFree(state_d));
  HANDLE_ERROR(cudaFree(x_d));
  HANDLE_ERROR(cudaFree(U_d));
  HANDLE_ERROR(cudaFree(du_d));
  HANDLE_ERROR(cudaFree(nu_d));
  HANDLE_ERROR(cudaFree(costs_d));
}
