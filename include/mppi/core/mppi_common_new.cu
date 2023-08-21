#include <mppi/utils/math_utils.h>
#include <mppi/core/mppi_common.cuh>

namespace mp1 = mppi::p1;

// CUDA barriers were first implemented in Cuda 11
#if defined(CUDART_VERSION) && CUDART_VERSION > 11000
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define USE_CUDA_BARRIERS_DYN
#define USE_CUDA_BARRIERS_COST
#endif

namespace mppi
{
namespace kernels
{
template <class COST_T, class SAMPLING_T, int BLOCKSIZE_X, bool COALESCE>
__global__ void rolloutCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                  const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                  const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;
  const int distribution_idx = threadIdx.z;
  const int size_of_theta_c_bytes =
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  int running_cost_index = thread_idx + blockDim.x * (thread_idy + blockDim.y * thread_idz);
  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];
  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::CONTROL_DIM)];
  int* crash_status_shared = (int*)&running_cost_shared[math::nearest_multiple_4(blockDim.x * blockDim.y * blockDim.z)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  float* theta_d = &theta_c[size_of_theta_c_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_COST
  barrier* barrier_shared = (barrier*)&theta_d[size_of_theta_d_bytes / sizeof(float)];
#endif

  // Create local state, state dot and controls
  float* y;
  float* u;
  int* crash_status;

  // Initialize running cost and total cost
  float* running_cost;
  int sample_time_offset = 0;
  int j = 0;

  // Load global array to shared array
  y = &y_shared[(blockDim.x * thread_idz + thread_idx) * COST_T::OUTPUT_DIM];
  u = &u_shared[(blockDim.x * thread_idz + thread_idx) * COST_T::CONTROL_DIM];
  crash_status = &crash_status_shared[thread_idz * blockDim.x + thread_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  // running_cost = &running_cost_shared[thread_idz * blockDim.x + thread_idx];
  running_cost = &running_cost_shared[running_cost_index];
  running_cost[0] = 0.0f;
#ifdef USE_CUDA_BARRIERS_COST
  barrier* bar = &barrier_shared[(blockDim.x * thread_idz + thread_idx)];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  /*<----Start of simulation loop-----> */
  const int max_time_iters = ceilf((float)num_timesteps / blockDim.x);
  costs->initializeCosts(y, u, theta_c, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x + 1;  // start at t = 1
    if (t <= num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      if (COALESCE)
      {  // Fill entire shared mem sequentially using sequential threads_idx
        mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_XY>(
            y_shared, blockDim.x * thread_idz * COST_T::OUTPUT_DIM, y_d,
            ((num_rollouts * thread_idz + global_idx) * num_timesteps + time_iter * blockDim.x) * COST_T::OUTPUT_DIM,
            COST_T::OUTPUT_DIM * blockDim.x);
      }
      else
      {
        sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t - 1;
        mp1::loadArrayParallel<COST_T::OUTPUT_DIM>(y, 0, y_d, sample_time_offset * COST_T::OUTPUT_DIM);
      }
    }
    if (t < num_timesteps)
    {  // load controls from t = 1 to t = num_timesteps - 1
      sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d, blockDim.y, thread_idy, y);
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // Compute cost
    if (t < num_timesteps)
    {
      running_cost[0] +=
          costs->computeRunningCost(y, u, t, theta_c, crash_status) +
          sampling->computeLikelihoodRatioCost(u, theta_d, global_idx, t, distribution_idx, lambda, alpha);
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
  }

  // Add all costs together
  running_cost = &running_cost_shared[blockDim.x * blockDim.y * thread_idz];
#if true
  int prev_size = blockDim.x * blockDim.y;
  // Allow for better computation when blockDim.x is a power of 2
  const bool block_power_of_2 = (prev_size & (prev_size - 1)) == 0;
  const int stop_condition = (block_power_of_2) ? 32 : 0;
  int size;
  const int xy_index = thread_idx + blockDim.x * thread_idy;
  const int xy_step = blockDim.x * blockDim.y;

  for (size = prev_size / 2; size > stop_condition; size /= 2)
  {
    for (j = xy_index; j < size; j += xy_step)
    {
      running_cost[j] += running_cost[j + size];
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1 && thread_idy == 0)
    {
      running_cost[size - 1] += running_cost[prev_size - 1];
    }
    __syncthreads();
    prev_size = size;
  }
  if (xy_index < 32)
  {  // unroll the last warp
    switch (size * 2)
    {
      case 64:
        warpReduceAdd<64>(running_cost, xy_index);
        break;
      case 32:
        warpReduceAdd<32>(running_cost, xy_index);
        break;
      case 16:
        warpReduceAdd<16>(running_cost, xy_index);
        break;
      case 8:
        warpReduceAdd<8>(running_cost, xy_index);
        break;
      case 4:
        warpReduceAdd<4>(running_cost, xy_index);
        break;
      case 2:
        warpReduceAdd<2>(running_cost, xy_index);
        break;
      case 1:
        warpReduceAdd<1>(running_cost, xy_index);
        break;
    }
  }
#else
  int prev_size = BLOCKSIZE_X;
#pragma unroll
  for (int size = prev_size / 2; size > 0; size /= 2)
  {
    if (thread_idy == 0)
    {
      for (j = thread_idx; j < size; j += blockDim.x)
      {
        running_cost[j] += running_cost[j + size];
      }
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1 && thread_idy == 0)
    {
      running_cost[size - 1] += running_cost[prev_size - 1];
    }
    __syncthreads();
    prev_size = size;
  }
#endif
  __syncthreads();
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % blockDim.x;
  y = &y_shared[(blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM];
  // Compute terminal cost and the final cost for each thread
  mppi_common::computeAndSaveCost(num_rollouts, num_timesteps, global_idx, costs, y,
                                  running_cost[0] / (num_timesteps - 1), theta_c, trajectory_costs_d);
}

template <class DYN_T, class SAMPLING_T>
__global__ void rolloutDynamicsKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling, float dt,
                                      const int num_timesteps, const int num_rollouts,
                                      const float* __restrict__ init_x_d, float* __restrict__ y_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int block_idx = blockIdx.x;
  const int global_idx = blockDim.x * block_idx + thread_idx;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int distribution_idx = threadIdx.z;
  const int distribution_dim = blockDim.z;
  const int sample_dim = blockDim.x;
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  const int size_of_theta_s_bytes =
      math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim * math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim *
          math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* x_shared = entire_buffer;
  float* x_next_shared = &x_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[math::nearest_multiple_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[math::nearest_multiple_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_DYN
  barrier* barrier_shared = (barrier*)&theta_d_shared[size_of_theta_d_bytes / sizeof(float)];
#endif

  // Create local state, state dot and controls
  float* x = &x_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_next = &x_next_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_temp;
  float* xdot = &x_dot_shared[shared_idx * DYN_T::STATE_DIM];
  float* u = &u_shared[shared_idx * DYN_T::CONTROL_DIM];
  float* y = &y_shared[shared_idx * DYN_T::OUTPUT_DIM];
#ifdef USE_CUDA_BARRIERS_DYN
  barrier* bar = &barrier_shared[shared_idx];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  // Load global array to shared array
  // const int blocksize_y = blockDim.y;
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz,
                                                           init_x_d, x, xdot, u);
  __syncthreads();

  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d_shared);
  __syncthreads();
  for (int t = 0; t < num_timesteps; t++)
  {
    // __syncthreads();
    // Load noise trajectories scaled by the exploration factor
    sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);
    // du_d is now v
    // __syncthreads();
    // if (global_idx == 0 && t == 0 && threadIdx.y == 0)
    // {
    //   printf("Control before: ");
    //   for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    //   {
    //     printf("%f, ", u[i]);
    //   }
    //   printf("\n");
    // }
#ifdef USE_CUDA_BARRIERS_DYN
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
    dynamics->enforceConstraints(x, u);
#ifdef USE_CUDA_BARRIERS_DYN
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    // Copy control constraints back to global memory
    sampling->writeControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
#ifdef USE_CUDA_BARRIERS_DYN
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    x_temp = x;
    x = x_next;
    x_next = x_temp;
    // Copy state to global memory
    int sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t;
    mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y_d, sample_time_offset * DYN_T::OUTPUT_DIM, y, 0);
  }
}

template <class COST_T, class SAMPLING_T, bool COALESCE>
__global__ void visualizeCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                    const int num_timesteps, const int num_rollouts, const float lambda, float alpha,
                                    const float* __restrict__ y_d, float* __restrict__ cost_traj_d,
                                    int* __restrict__ crash_status_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  // const int block_idx = blockIdx.x;
  const int global_idx = blockIdx.x;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int distribution_idx = threadIdx.z;
  // const int distribution_dim = blockDim.z;
  // const int sample_dim = blockDim.x;
  const int size_of_theta_c_bytes =
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::CONTROL_DIM)];
  int* crash_status_shared =
      (int*)&running_cost_shared[math::nearest_multiple_4(blockDim.y * blockDim.z * num_timesteps)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  float* theta_d = &theta_c[size_of_theta_c_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_COST
  barrier* barrier_shared = (barrier*)&theta_d[size_of_theta_d_bytes / sizeof(float)];
#endif
  // Create local state, state dot and controls
  float* y;
  float* u;
  // float* du;
  int* crash_status;

  // Initialize running cost and total cost
  float* running_cost;
  int sample_time_offset = 0;
  int cost_index = 0;

  // Load global array to shared array
  y = &y_shared[shared_idx * COST_T::OUTPUT_DIM];
  u = &u_shared[shared_idx * COST_T::CONTROL_DIM];
  // du = &du_shared[shared_idx * COST_T::CONTROL_DIM];
  crash_status = &crash_status_shared[shared_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
#ifdef USE_CUDA_BARRIERS_COST
  barrier* bar = &barrier_shared[(blockDim.x * thread_idz + thread_idx)];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif
  // running_cost = &running_cost_shared[shared_idx];
  // running_cost[0] = 0;

  /*<----Start of simulation loop-----> */
  const int max_time_iters = ceilf((float)num_timesteps / blockDim.x);
  costs->initializeCosts(y, u, theta_c, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x + 1;
    cost_index = (thread_idz * num_rollouts + global_idx) * (num_timesteps) + t - 1;
    running_cost = &running_cost_shared[blockDim.x * (thread_idz * blockDim.y + thread_idy) + t - 1];
    if (t <= num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      if (COALESCE)
      {  // Fill entire shared mem sequentially using sequential threads_idx
        mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_XY>(
            y_shared, blockDim.x * thread_idz, y_d,
            ((num_rollouts * thread_idz + global_idx) * num_timesteps + time_iter * blockDim.x) * COST_T::OUTPUT_DIM,
            COST_T::OUTPUT_DIM * blockDim.x);
      }
      else
      {
        sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t - 1;
        mp1::loadArrayParallel<COST_T::OUTPUT_DIM>(y, 0, y_d, sample_time_offset * COST_T::OUTPUT_DIM);
      }
    }
    if (t < num_timesteps)
    {  // load controls from t = 1 to t = num_timesteps - 1
      sampling->readVisControlSample(global_idx, t, distribution_idx, u, theta_d, blockDim.y, thread_idy, y);
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // Compute cost
    if (t < num_timesteps)
    {
      float cost = costs->computeRunningCost(y, u, t, theta_c, crash_status) +
                   sampling->computeLikelihoodRatioCost(u, theta_d, global_idx, t, distribution_idx, lambda, alpha);
      running_cost[0] = cost / (num_timesteps - 1);
      crash_status_d[global_idx * num_timesteps + t] = crash_status[0];
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
  }
  // consolidate y threads into single cost
  running_cost = &running_cost_shared[thread_idx + blockDim.x * blockDim.y * thread_idz];
  int prev_size = blockDim.y;
  int size;
  int j;
  for (size = prev_size / 2; size > 0; size /= 2)
  {
    for (j = thread_idy; j < size; j += blockDim.y)
    {
      running_cost[j * blockDim.x] += running_cost[(j + size) * blockDim.x];
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.y == blockDim.y - 1)
    {
      running_cost[(size - 1) * blockDim.x] += running_cost[(prev_size - 1) * blockDim.x];
    }
    __syncthreads();
    prev_size = size;
  }
  __syncthreads();
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % blockDim.x;
  y = &y_shared[(blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM];
  // Compute terminal cost for each thread
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    cost_index = (threadIdx.z * num_rollouts + global_idx) * (num_timesteps + 1) + num_timesteps;
    cost_traj_d[cost_index] = costs->terminalCost(y, theta_c) / (num_timesteps - 1);
    // running_cost = &running_cost_shared[thread_idz * blockDim.x + num_timesteps];
    // running_cost[0] = costs->terminalCost(y, theta_c) / (num_timesteps - 1);
  }
  __syncthreads();
  // Copy to global memory
  if (num_timesteps % 4 == 0)
  {
    float4* cost_traj_d4 =
        reinterpret_cast<float4*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps]);
    float4* running_cost_shared4 =
        reinterpret_cast<float4*>(&running_cost_shared[thread_idz * num_timesteps * blockDim.y]);
    for (int i = thread_idx; i < num_timesteps / 4; i += blockDim.x)
    {
      cost_traj_d4[i] = running_cost_shared4[i];
    }
  }
  else if (num_timesteps % 2 == 0)
  {
    float2* cost_traj_d2 =
        reinterpret_cast<float2*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps]);
    float2* running_cost_shared2 =
        reinterpret_cast<float2*>(&running_cost_shared[thread_idz * num_timesteps * blockDim.y]);
    for (int i = thread_idx; i < num_timesteps / 2; i += blockDim.x)
    {
      cost_traj_d2[i] = running_cost_shared2[i];
    }
  }
  else
  {
    for (int i = thread_idx; i < num_timesteps; i += blockDim.x)
    {
      cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps + i] =
          running_cost_shared[thread_idz * num_timesteps * blockDim.y + i];
    }
  }
}

template <int CONTROL_DIM>
__global__ void weightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                        float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                        const int num_rollouts, const int sum_stride)
{
  int thread_idx = threadIdx.x;  // Rollout index
  int block_idx = blockIdx.x;    // Timestep

  // Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
  extern __shared__ float u_intermediate[];

  float u[CONTROL_DIM];
  setInitialControlToZero(CONTROL_DIM, thread_idx, u, u_intermediate);

  __syncthreads();

  // Sum the weighted control variations at a desired stride
  strideControlWeightReduction(num_rollouts, num_timesteps, sum_stride, thread_idx, block_idx, CONTROL_DIM, exp_costs_d,
                               normalizer, du_d, u, u_intermediate);

  __syncthreads();

  // Sum all weighted control variations
  mppi_common::rolloutWeightReductionAndSaveControl(thread_idx, block_idx, num_rollouts, num_timesteps, CONTROL_DIM,
                                                    sum_stride, u, u_intermediate, new_u_d);

  __syncthreads();
}

template <int STATE_DIM, int CONTROL_DIM>
__device__ void loadGlobalToShared(const int num_rollouts, const int blocksize_y, const int global_idx,
                                   const int thread_idy, const int thread_idz, const float* __restrict__ x_device,
                                   float* __restrict__ x_thread, float* __restrict__ xdot_thread,
                                   float* __restrict__ u_thread)
{
  // Transfer to shared memory
  int i;
  // float zero_state[STATE_DIM] = { 0 };
  if (global_idx < num_rollouts)
  {
#if true
    mp1::loadArrayParallel<STATE_DIM>(x_thread, 0, x_device, STATE_DIM * thread_idz);
    if (STATE_DIM % 4 == 0)
    {
      float4* xdot4_t = reinterpret_cast<float4*>(xdot_thread);
      for (i = thread_idy; i < STATE_DIM / 4; i += blocksize_y)
      {
        xdot4_t[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
    else if (STATE_DIM % 2 == 0)
    {
      float2* xdot2_t = reinterpret_cast<float2*>(xdot_thread);
      for (i = thread_idy; i < STATE_DIM / 2; i += blocksize_y)
      {
        xdot2_t[i] = make_float2(0.0f, 0.0f);
      }
    }
    else
    {
      for (i = thread_idy; i < STATE_DIM; i += blocksize_y)
      {
        xdot_thread[i] = 0.0f;
      }
    }

    if (CONTROL_DIM % 4 == 0)
    {
      float4* u4_t = reinterpret_cast<float4*>(u_thread);
      for (i = thread_idy; i < CONTROL_DIM / 4; i += blocksize_y)
      {
        u4_t[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
    else if (CONTROL_DIM % 2 == 0)
    {
      float2* u2_t = reinterpret_cast<float2*>(u_thread);
      for (i = thread_idy; i < CONTROL_DIM / 2; i += blocksize_y)
      {
        u2_t[i] = make_float2(0.0f, 0.0f);
      }
    }
    else
    {
      for (i = thread_idy; i < CONTROL_DIM; i += blocksize_y)
      {
        u_thread[i] = 0.0f;
      }
    }
#else
    for (i = thread_idy; i < STATE_DIM; i += blocksize_y)
    {
      x_thread[i] = x_device[i + STATE_DIM * thread_idz];
      xdot_thread[i] = 0.0f;
    }
    for (i = thread_idy; i < CONTROL_DIM; i += blocksize_y)
    {
      u_thread[i] = 0.0f;
    }
#endif
  }
}

__device__ void strideControlWeightReduction(const int num_rollouts, const int num_timesteps, const int sum_stride,
                                             const int thread_idx, const int block_idx, const int control_dim,
                                             const float* __restrict__ exp_costs_d, const float normalizer,
                                             const float* __restrict__ du_d, float* __restrict__ u,
                                             float* __restrict__ u_intermediate)
{
  // int index = thread_idx * sum_stride + i;
  for (int i = 0; i < sum_stride; ++i)
  {  // Iterate through the size of the subsection
    if ((thread_idx * sum_stride + i) < num_rollouts)
    {                                                                        // Ensure we do not go out of bounds
      float weight = exp_costs_d[thread_idx * sum_stride + i] / normalizer;  // compute the importance sampling weight
      for (int j = 0; j < control_dim; ++j)
      {  // Iterate through the control dimensions
        // Rollout index: (thread_idx*sum_stride + i)*(num_timesteps*control_dim)
        // Current timestep: block_idx*control_dim
        u[j] = du_d[(thread_idx * sum_stride + i) * (num_timesteps * control_dim) + block_idx * control_dim + j];
        u_intermediate[thread_idx * control_dim + j] += weight * u[j];
      }
    }
  }
}

template <int BLOCKSIZE>
__device__ void warpReduceAdd(volatile float* sdata, const int tid, const int stride)
{
  if (BLOCKSIZE >= 64)
  {
    sdata[tid * stride] += sdata[(tid + 32) * stride];
  }
  if (BLOCKSIZE >= 32)
  {
    sdata[tid * stride] += sdata[(tid + 16) * stride];
  }
  if (BLOCKSIZE >= 16)
  {
    sdata[tid * stride] += sdata[(tid + 8) * stride];
  }
  if (BLOCKSIZE >= 8)
  {
    sdata[tid * stride] += sdata[(tid + 4) * stride];
  }
  if (BLOCKSIZE >= 4)
  {
    sdata[tid * stride] += sdata[(tid + 2) * stride];
  }
  if (BLOCKSIZE >= 2)
  {
    sdata[tid * stride] += sdata[(tid + 1) * stride];
  }
}

template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchFastRolloutKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs,
                             SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                             const int num_rollouts, float lambda, float alpha, float* __restrict__ init_x_d,
                             float* __restrict__ y_d, float* __restrict__ trajectory_costs, dim3 dimDynBlock,
                             dim3 dimCostBlock, cudaStream_t stream, bool synchronize)
{
  if (num_rollouts % dimDynBlock.x != 0)
  {
    std::cerr << __FILE__ << " (" << __LINE__ << "): num_rollouts (" << num_rollouts
              << ") must be evenly divided by dynamics block size x (" << dimDynBlock.x << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (num_timesteps < dimCostBlock.x)
  {
    std::cerr << __FILE__ << " (" << __LINE__ << "): num_timesteps (" << num_timesteps
              << ") must be greater than or equal to cost block size x (" << dimCostBlock.x << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Run Dynamics
  const int gridsize_x = math::int_ceil(num_rollouts, dimDynBlock.x);
  dim3 dimGrid(gridsize_x, 1, 1);
  const int dynamics_num_shared = dimDynBlock.x * dimDynBlock.z;
  unsigned dynamics_shared_size =
      sizeof(float) * (3 * math::nearest_multiple_4(dynamics_num_shared * DYN_T::STATE_DIM) +
                       math::nearest_multiple_4(dynamics_num_shared * DYN_T::OUTPUT_DIM) +
                       math::nearest_multiple_4(dynamics_num_shared * DYN_T::CONTROL_DIM)) +
      math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
#ifdef USE_CUDA_BARRIERS_DYN
  dynamics_shared_size += math::int_multiple_const(dynamics_num_shared * sizeof(barrier), 16);
#endif
  // std::cout << "Grid: (" << dimGrid.x << ", " << dimGrid.y << ", " << dimGrid.z << "), Block: ("
  //           << dimDynBlock.x << ", " << dimDynBlock.y << ", " << dimDynBlock.z << "), shared_mem: "
  //           << dynamics_shared_size << " bytes" << std::endl;
  // std::cout << "Shared mem size for sampling: " <<
  //     math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
  //     dynamics_num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4))
  //     << " GRD_BYTES: " << SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES << " BLK BYTES: " <<
  //     SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES
  //     << std::endl;
  rolloutDynamicsKernel<DYN_T, SAMPLING_T><<<dimGrid, dimDynBlock, dynamics_shared_size, stream>>>(
      dynamics, sampling, dt, num_timesteps, num_rollouts, init_x_d, y_d);
  // Run Costs
  dim3 dimCostGrid(num_rollouts, 1, 1);
  const int cost_num_shared = dimCostBlock.x * dimCostBlock.z;
  const int COST_BLOCK_X = 64;
  unsigned cost_shared_size =
      sizeof(float) * (math::nearest_multiple_4(cost_num_shared * COST_T::OUTPUT_DIM) +
                       math::nearest_multiple_4(cost_num_shared * COST_T::CONTROL_DIM) +
                       math::nearest_multiple_4(cost_num_shared * dimCostBlock.y)) +
      sizeof(int) * math::nearest_multiple_4(cost_num_shared) +
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      cost_num_shared * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      cost_num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
#ifdef USE_CUDA_BARRIERS_COST
  cost_shared_size += math::int_multiple_const(cost_num_shared * sizeof(barrier), 16);
#endif
  rolloutCostKernel<COST_T, SAMPLING_T, COST_BLOCK_X><<<dimCostGrid, dimCostBlock, cost_shared_size, stream>>>(
      costs, sampling, dt, num_timesteps, num_rollouts, lambda, alpha, y_d, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class COST_T, class SAMPLING_T>
void launchVisualizeCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                               const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                               float* __restrict__ y_d, int* __restrict__ sampled_crash_status_d,
                               float* __restrict__ cost_traj_result, dim3 dimBlock, cudaStream_t stream,
                               bool synchronize)
{
  if (num_rollouts <= 1)
  {  // Not enough samples to visualize
    std::cerr << "Not enough samples to visualize" << std::endl;
    return;
  }
  // Run Costs
  dim3 dimCostGrid(num_rollouts, 1, 1);

  const int shared_num = dimBlock.x * dimBlock.z;
  unsigned shared_mem_size =
      sizeof(float) * (math::nearest_multiple_4(shared_num * COST_T::OUTPUT_DIM) +
                       math::nearest_multiple_4(shared_num * COST_T::CONTROL_DIM) +
                       math::nearest_multiple_4(dimBlock.z * num_timesteps * dimBlock.y)) +
      sizeof(int) * math::nearest_multiple_4(dimBlock.z * num_timesteps) +
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      shared_num * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      shared_num * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
#ifdef USE_CUDA_BARRIERS_COST
  shared_mem_size += math::int_multiple_const(shared_num * sizeof(barrier), 16);
#endif
  // unsigned shared_mem_size =
  //     ((COST_BLOCK_X * BLOCKSIZE_Z) * (COST_T::OUTPUT_DIM + 2 * COST_T::CONTROL_DIM) + COST_T::CONTROL_DIM) *
  //         sizeof(float) +
  //     sizeof(float) * (num_timesteps * BLOCKSIZE_Z) + (COST_BLOCK_X * BLOCKSIZE_Z) * sizeof(int) +
  //     COST_T::SHARED_MEM_REQUEST_GRD_BYTES + sizeof(float) +
  //     COST_T::SHARED_MEM_REQUEST_BLK_BYTES * COST_BLOCK_X * BLOCKSIZE_Z;
  visualizeCostKernel<COST_T, SAMPLING_T><<<dimCostGrid, dimBlock, shared_mem_size, stream>>>(
      costs, sampling, dt, num_timesteps, num_rollouts, lambda, alpha, y_d, cost_traj_result, sampled_crash_status_d);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <int CONTROL_DIM>
void launchWeightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                   float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                   const int num_rollouts, const int sum_stride, cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock(math::int_ceil(num_rollouts, sum_stride), 1, 1);
  dim3 dimGrid(num_timesteps, 1, 1);
  unsigned shared_mem_size = math::nearest_multiple_4(CONTROL_DIM * dimBlock.x) * sizeof(float);
  weightedReductionKernel<CONTROL_DIM><<<dimGrid, dimBlock, shared_mem_size, stream>>>(
      exp_costs_d, du_d, new_u_d, normalizer, num_timesteps, num_rollouts, sum_stride);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

__device__ void setInitialControlToZero(int control_dim, int thread_idx, float* __restrict__ u,
                                        float* __restrict__ u_intermediate)
{
  if (control_dim % 4 == 0)
  {
    for (int i = 0; i < control_dim / 4; i++)
    {
      reinterpret_cast<float4*>(u)[i] = make_float4(0, 0, 0, 0);
      reinterpret_cast<float4*>(&u_intermediate[thread_idx * control_dim])[i] = make_float4(0, 0, 0, 0);
    }
  }
  else if (control_dim % 2 == 0)
  {
    for (int i = 0; i < control_dim / 2; i++)
    {
      reinterpret_cast<float2*>(u)[i] = make_float2(0, 0);
      reinterpret_cast<float2*>(&u_intermediate[thread_idx * control_dim])[i] = make_float2(0, 0);
    }
  }
  else
  {
    for (int i = 0; i < control_dim; i++)
    {
      u[i] = 0;
      u_intermediate[thread_idx * control_dim + i] = 0;
    }
  }
}

}  // namespace kernels
}  // namespace mppi
