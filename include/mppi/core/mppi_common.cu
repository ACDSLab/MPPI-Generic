#include <mppi/core/mppi_common.cuh>
#include <curand.h>
#include <mppi/utils/gpu_err_chk.cuh>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/cuda_math_utils.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
namespace mp1 = mppi::p1;

namespace mppi_common
{
/*******************************************************************************************************************
 * Kernel Functions
 *******************************************************************************************************************/
// TODO remove dt
template <class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int NUM_ROLLOUTS, int BLOCKSIZE_Z>
__global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, int optimization_stride,
                              float lambda, float alpha, float* x_d, float* u_d, float* du_d, float* sigma_u_d,
                              float* trajectory_costs_d)
{
  // Get thread and block id
  int thread_idx = threadIdx.x;
  int thread_idy = threadIdx.y;
  int thread_idz = threadIdx.z;
  int block_idx = blockIdx.x;
  int global_idx = BLOCKSIZE_X * block_idx + thread_idx;

  // Create shared state and control arrays
  __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float x_next_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float y_shared[BLOCKSIZE_X * DYN_T::OUTPUT_DIM * BLOCKSIZE_Z];
  __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float sigma_u[DYN_T::CONTROL_DIM];
  __shared__ int crash_status_shared[BLOCKSIZE_X * BLOCKSIZE_Z];

  // Create a shared array for the dynamics model to use
  __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 +
                           DYN_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X * BLOCKSIZE_Z];
  __shared__ float theta_c[COST_T::SHARED_MEM_REQUEST_GRD + COST_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X * BLOCKSIZE_Z];

  // Create local state, state dot and controls
  float* x;
  float* x_next;
  float* x_temp;
  float* y;
  float* xdot;
  float* u;
  float* du;
  int* crash_status;

  // Initialize running cost and total cost
  float running_cost = 0;
  // Load global array to shared array
  if (global_idx < NUM_ROLLOUTS)
  {
    x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    x_next = &x_next_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    y = &y_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::OUTPUT_DIM];
    xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    crash_status = &crash_status_shared[thread_idz * blockDim.x + thread_idx];
    crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  }
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(NUM_ROLLOUTS, BLOCKSIZE_Y, global_idx, thread_idy,
                                                           thread_idz, x_d, sigma_u_d, x, xdot, u, du, sigma_u);
  __syncthreads();

  if (global_idx < NUM_ROLLOUTS)
  {
    /*<----Start of simulation loop-----> */
    dynamics->initializeDynamics(x, u, y, theta_s, 0.0, dt);
    costs->initializeCosts(y, u, theta_c, 0.0, dt);
    __syncthreads();
    for (int t = 0; t < num_timesteps; t++)
    {
      // Load noise trajectories scaled by the exploration factor
      injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps, t, global_idx, thread_idy,
                         optimization_stride, u_d, du_d, sigma_u, u, du);
      // du_d is now v
      __syncthreads();

      // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
      // usually just control clamping
      // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
      dynamics->enforceConstraints(x, &du_d[(NUM_ROLLOUTS * num_timesteps * threadIdx.z +  // z part
                                             global_idx * num_timesteps + t) *
                                            DYN_T::CONTROL_DIM]);
      dynamics->enforceConstraints(x, u);
      __syncthreads();

      // Accumulate running cost
      if (thread_idy == 0 && t > 0)
      {
        running_cost +=
            (costs->computeRunningCost(y, u, du, sigma_u, lambda, alpha, t, theta_c, crash_status) - running_cost) /
            (t);
        // running_cost +=
        //     costs->computeRunningCost(y, u, du, sigma_u, lambda, alpha, t, theta_c, crash_status) / (num_timesteps -
        //     1);
      }

      // Compute state derivatives
      __syncthreads();  // Ensure step doesn't change y while cost above is caclulating
      dynamics->step(x, x_next, xdot, u, y, theta_s, t, dt);
      __syncthreads();
      x_temp = x;
      x = x_next;
      x_next = x_temp;
    }
    // Compute terminal cost and the final cost for each thread
    computeAndSaveCost(NUM_ROLLOUTS, num_timesteps, global_idx, costs, y, running_cost, theta_c, trajectory_costs_d);
  }
}

template <class DYN_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int NUM_ROLLOUTS, int BLOCKSIZE_Z>
__global__ void rolloutDynamicsKernel(DYN_T* __restrict__ dynamics, float dt, int num_timesteps,
                                      int optimization_stride, const float* __restrict__ init_x_d,
                                      const float* __restrict__ u_d, float* __restrict__ du_d,
                                      const float* __restrict__ sigma_u_d, float* __restrict__ y_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int block_idx = blockIdx.x;
  const int global_idx = BLOCKSIZE_X * block_idx + thread_idx;

  // Create shared state and control arrays
  __shared__ float4 x_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z, 4)];
  __shared__ float4 x_next_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z, 4)];
  __shared__ float4 y_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::OUTPUT_DIM * BLOCKSIZE_Z, 4)];
  __shared__ float4 xdot_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z, 4)];
  __shared__ float4 u_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z, 4)];
  __shared__ float4 du_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z, 4)];
  __shared__ float4 sigma_u[mppi::math::int_ceil(DYN_T::CONTROL_DIM, 4)];

  // Create a shared array for the dynamics model to use
  __shared__ float4 theta_s4[mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 +
                                                      DYN_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X * BLOCKSIZE_Z,
                                                  4)];

  float* theta_s = reinterpret_cast<float*>(theta_s4);
  // Create local state, state dot and controls
  float* x;
  float* x_next;
  float* x_temp;
  float* xdot;
  float* u;
  float* du;
  float* y;

  // Load global array to shared array
  if (global_idx < NUM_ROLLOUTS)
  {
    x = &(reinterpret_cast<float*>(x_shared)[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM]);
    x_next = &(reinterpret_cast<float*>(x_next_shared)[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM]);
    y = &(reinterpret_cast<float*>(y_shared)[(blockDim.x * thread_idz + thread_idx) * DYN_T::OUTPUT_DIM]);
    xdot = &(reinterpret_cast<float*>(xdot_shared)[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM]);
    u = &(reinterpret_cast<float*>(u_shared)[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM]);
    du = &(reinterpret_cast<float*>(du_shared)[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM]);
  }
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(NUM_ROLLOUTS, BLOCKSIZE_Y, global_idx, thread_idy,
                                                           thread_idz, init_x_d, sigma_u_d, x, xdot, u, du,
                                                           reinterpret_cast<float*>(sigma_u));
  __syncthreads();

  if (global_idx < NUM_ROLLOUTS)
  {
    /*<----Start of simulation loop-----> */
    dynamics->initializeDynamics(x, u, y, theta_s, 0.0, dt);
    for (int t = 0; t < num_timesteps; t++)
    {
      // Load noise trajectories scaled by the exploration factor
      injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps, t, global_idx, thread_idy,
                         optimization_stride, u_d, du_d, reinterpret_cast<float*>(sigma_u), u, du);
      // du_d is now v
      __syncthreads();

      // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
      // usually just control clamping
      // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
      dynamics->enforceConstraints(x, &du_d[(NUM_ROLLOUTS * num_timesteps * threadIdx.z +  // z part
                                             global_idx * num_timesteps + t) *
                                            DYN_T::CONTROL_DIM]);
      dynamics->enforceConstraints(x, u);
      __syncthreads();

      // Increment states
      dynamics->step(x, x_next, xdot, u, y, theta_s, t, dt);
      __syncthreads();
      x_temp = x;
      x = x_next;
      x_next = x_temp;
      // Copy state to global memory
      int sample_time_offset = (NUM_ROLLOUTS * thread_idz + global_idx) * num_timesteps + t;
      mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y_d, sample_time_offset * DYN_T::OUTPUT_DIM, y, 0);
    }
  }
}

template <class DYN_T, class COST_T, int NUM_ROLLOUTS, int BLOCKSIZE_X, bool COALESCE = false>
__global__ void rolloutCostKernel(DYN_T* dynamics, COST_T* costs, float dt, const int num_timesteps, float lambda,
                                  float alpha, const float* __restrict__ init_x_d, const float* __restrict__ u_d,
                                  const float* __restrict__ du_d, const float* __restrict__ sigma_u_d,
                                  const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];
  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[blockDim.x * blockDim.z * DYN_T::OUTPUT_DIM];
  float* du_shared = &u_shared[blockDim.x * blockDim.z * DYN_T::CONTROL_DIM];
  float* sigma_u = &du_shared[blockDim.x * blockDim.z * DYN_T::CONTROL_DIM];
  float* running_cost_shared = &sigma_u[DYN_T::CONTROL_DIM];
  int* crash_status_shared = (int*)&running_cost_shared[blockDim.x * blockDim.z];
  float* theta_c = (float*)&crash_status_shared[blockDim.x * blockDim.z];

  // Create local state, state dot and controls
  float* y;
  float* u;
  float* du;
  int* crash_status;

  // Initialize running cost and total cost
  float* running_cost;
  int sample_time_offset = 0;
  int j = 0;

  // Load global array to shared array
  y = &y_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::OUTPUT_DIM];
  u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
  du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
  crash_status = &crash_status_shared[thread_idz * blockDim.x + thread_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  running_cost = &running_cost_shared[thread_idz * blockDim.x + thread_idx];
  running_cost[0] = 0;
  if (thread_idx == 0)
  {
    mp1::loadArrayParallel<DYN_T::CONTROL_DIM>(sigma_u, 0, sigma_u_d, 0);
  }

  /*<----Start of simulation loop-----> */
  const int max_time_iters = ceilf((float)num_timesteps / BLOCKSIZE_X);
  costs->initializeCosts(y, u, theta_c, 0.0, dt);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x + 1;
    if (t <= num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      if (COALESCE)
      {  // Fill entire shared mem sequentially using sequential threads_idx
        mp1::loadArrayParallel<DYN_T::OUTPUT_DIM * BLOCKSIZE_X, mp1::Parallel1Dir::THREAD_X>(
            y_shared, blockDim.x * thread_idz, y_d,
            ((NUM_ROLLOUTS * thread_idz + global_idx) * num_timesteps + time_iter * blockDim.x) * DYN_T::OUTPUT_DIM);
      }
      else
      {
        sample_time_offset = (NUM_ROLLOUTS * thread_idz + global_idx) * num_timesteps + t - 1;
        mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y, 0, y_d, sample_time_offset * DYN_T::OUTPUT_DIM);
      }
    }
    if (t < num_timesteps)
    {  // load controls from t = 0 to t = num_timesteps - 1
      // Have to do similar steps as injectControlNoise but using the already transformed cost samples
      readControlsFromGlobal(DYN_T::CONTROL_DIM, blockDim.y, NUM_ROLLOUTS, num_timesteps, t, global_idx, thread_idy,
                             u_d, du_d, u, du);
    }
    __syncthreads();

    // dynamics->enforceConstraints(x, u);
    // __syncthreads();
    // Compute cost
    if (thread_idy == 0 && t < num_timesteps)
    {
      running_cost[0] += costs->computeRunningCost(y, u, du, sigma_u, lambda, alpha, t, theta_c, crash_status);
    }
    __syncthreads();
  }

  // Add all costs together
  int prev_size = BLOCKSIZE_X;
  running_cost = &running_cost_shared[blockDim.x * thread_idz];
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
  __syncthreads();
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % BLOCKSIZE_X;
  y = &y_shared[(blockDim.x * thread_idz + last_y_index) * DYN_T::OUTPUT_DIM];
  // Compute terminal cost and the final cost for each thread
  computeAndSaveCost(NUM_ROLLOUTS, num_timesteps, global_idx, costs, y, running_cost[0] / (num_timesteps - 1), theta_c,
                     trajectory_costs_d);
}

__global__ void normExpKernel(int num_rollouts, float* trajectory_costs_d, float lambda_inv, float baseline)
{
  int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * blockDim.z + threadIdx.z;
  int global_step = blockDim.x * gridDim.x * blockDim.z * gridDim.z;
  // #if defined(CUDA_VERSION) && CUDA_VERSION > 11060
  //   auto block = cg::this_grid();
  //   int global_idx_b = block.thread_rank() + block.block_rank() * block.num_threads();
  //   int global_step_b = block.num_threads() * block.num_blocks();
  //   if (global_idx == 200 && threadIdx.y == 0 && threadIdx.z == 0)
  //   {
  //     printf("Global ind: %d, thread_rank: %d\n", global_idx, global_idx_b);
  //     printf("Global step: %d, thread_rank: %d\n", global_step, global_step_b);
  //   }
  // #endif
  normExpTransform(num_rollouts * blockDim.z, trajectory_costs_d, lambda_inv, baseline, global_idx, global_step);
}

__global__ void TsallisKernel(int num_rollouts, float* trajectory_costs_d, float gamma, float r, float baseline)
{
  int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * blockDim.z + threadIdx.z;
  int global_step = blockDim.x * gridDim.x * blockDim.z * gridDim.z;
  TsallisTransform(num_rollouts * blockDim.z, trajectory_costs_d, gamma, r, baseline, global_idx, global_step);
}

template <int CONTROL_DIM, int NUM_ROLLOUTS, int SUM_STRIDE>
__global__ void weightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d, float normalizer,
                                        int num_timesteps)
{
  int thread_idx = threadIdx.x;  // Rollout index
  int block_idx = blockIdx.x;    // Timestep

  // Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
  __shared__ float u_intermediate[CONTROL_DIM * ((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1)];

  float u[CONTROL_DIM];
  setInitialControlToZero(CONTROL_DIM, thread_idx, u, u_intermediate);

  __syncthreads();

  // Sum the weighted control variations at a desired stride
  strideControlWeightReduction(NUM_ROLLOUTS, num_timesteps, SUM_STRIDE, thread_idx, block_idx, CONTROL_DIM, exp_costs_d,
                               normalizer, du_d, u, u_intermediate);

  __syncthreads();

  // Sum all weighted control variations
  rolloutWeightReductionAndSaveControl(thread_idx, block_idx, NUM_ROLLOUTS, num_timesteps, CONTROL_DIM, SUM_STRIDE, u,
                                       u_intermediate, du_new_d);

  __syncthreads();
}

template <int CONTROL_DIM, int NUM_ROLLOUTS, int SUM_STRIDE>
__global__ void weightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d,
                                        float2* baseline_and_normalizer_d, int num_timesteps)
{
  int thread_idx = threadIdx.x;  // Rollout index
  int block_idx = blockIdx.x;    // Timestep

  // Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
  __shared__ float u_intermediate[CONTROL_DIM * ((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1)];

  float u[CONTROL_DIM];
  setInitialControlToZero(CONTROL_DIM, thread_idx, u, u_intermediate);

  __syncthreads();

  // Sum the weighted control variations at a desired stride
  strideControlWeightReduction(NUM_ROLLOUTS, num_timesteps, SUM_STRIDE, thread_idx, block_idx, CONTROL_DIM, exp_costs_d,
                               baseline_and_normalizer_d->y, du_d, u, u_intermediate);

  __syncthreads();

  // Sum all weighted control variations
  rolloutWeightReductionAndSaveControl(thread_idx, block_idx, NUM_ROLLOUTS, num_timesteps, CONTROL_DIM, SUM_STRIDE, u,
                                       u_intermediate, du_new_d);

  __syncthreads();
}

/*******************************************************************************************************************
 * Rollout Kernel Helpers
 *******************************************************************************************************************/
template <int STATE_DIM, int CONTROL_DIM>
__device__ void loadGlobalToShared(const int num_rollouts, const int blocksize_y, const int global_idx,
                                   const int thread_idy, const int thread_idz, const float* x_device,
                                   const float* sigma_u_device, float* x_thread, float* xdot_thread, float* u_thread,
                                   float* du_thread, float* sigma_u_thread)
{
  // Transfer to shared memory
  int i;
  // float zero_state[STATE_DIM] = { 0 };
  if (global_idx < num_rollouts)
  {
#if false
    mp1::loadArrayParallel<STATE_DIM>(x_thread, 0, x_device, STATE_DIM * thread_idz);
    if (STATE_DIM % 4 == 0)
    {
      float4* xdot4_t = reinterpret_cast<float4*>(xdot_thread);
      for (i = thread_idy; i < STATE_DIM / 4; i +=blocksize_y)
      {
        xdot4_t[i] = make_float4(0, 0, 0, 0);
      }
    }
    else if (STATE_DIM % 2 == 0)
    {
      float2* xdot2_t = reinterpret_cast<float2*>(xdot_thread);
      for (i = thread_idy; i < STATE_DIM / 2; i +=blocksize_y)
      {
        xdot2_t[i] = make_float2(0, 0);
      }
    }
    else
    {
      for (i = thread_idy; i < STATE_DIM; i +=blocksize_y)
      {
        xdot_thread[i] = 0;
      }
    }

    if (CONTROL_DIM % 4 == 0)
    {
      float4* u4_t = reinterpret_cast<float4*>(u_thread);
      float4* du4_t = reinterpret_cast<float4*>(du_thread);
      for (i = thread_idy; i < CONTROL_DIM / 4; i +=blocksize_y)
      {
        u4_t[i] = make_float4(0, 0, 0, 0);
        du4_t[i] = make_float4(0, 0, 0, 0);
      }
    }
    else if (CONTROL_DIM % 2 == 0)
    {
      float2* u2_t = reinterpret_cast<float2*>(u_thread);
      float2* du2_t = reinterpret_cast<float2*>(du_thread);
      for (i = thread_idy; i < CONTROL_DIM / 2; i +=blocksize_y)
      {
        u2_t[i] = make_float2(0, 0);
        du2_t[i] = make_float2(0, 0);
      }
    }
    else
    {
      for (i = thread_idy; i < CONTROL_DIM; i += blocksize_y)
      {
        u_thread[i] = 0;
        du_thread[i] = 0;
      }
    }
#else
    for (i = thread_idy; i < STATE_DIM; i += blocksize_y)
    {
      x_thread[i] = x_device[i + STATE_DIM * thread_idz];
      xdot_thread[i] = 0;
    }
    for (i = thread_idy; i < CONTROL_DIM; i += blocksize_y)
    {
      u_thread[i] = 0;
      du_thread[i] = 0;
    }
#endif
  }
  if (threadIdx.x == 0 /*&& threadIdx.z == 0*/)
  {
    mp1::loadArrayParallel<CONTROL_DIM>(sigma_u_thread, 0, sigma_u_device, 0);
    // for (i = thread_idy; i < control_dim; i += blocksize_y)
    // {
    //   sigma_u_thread[i] = sigma_u_device[i];
    // }
  }
}

__device__ void readControlsFromGlobal(const int control_dim, const int blocksize_y, const int num_rollouts,
                                       const int num_timesteps, const int t, const int global_idx, const int thread_idy,
                                       const float* u_d, const float* du_d, float* u_thread, float* du_thread)
{
  const int control_index = ((num_rollouts * threadIdx.z + global_idx) * num_timesteps + t) * control_dim;
  if (control_dim % 4 == 0)
  {
    float4* du4 = reinterpret_cast<float4*>(du_thread);
    float4* u4 = reinterpret_cast<float4*>(u_thread);
    const float4* u4_mean_d = reinterpret_cast<const float4*>(&u_d[t * control_dim]);
    const float4* du4_d = reinterpret_cast<const float4*>(&du_d[control_index]);
    for (int j = thread_idy; j < control_dim / 4; j += blocksize_y)
    {
      if (global_idx == 0)
      {
        du4[j] = make_float4(0, 0, 0, 0);
        u4[j] = u4_mean_d[j];
      }
      else if (global_idx >= 0.99 * num_rollouts)
      {
        du4[j] = du4_d[j];
        u4[j] = du4[j];
      }
      else
      {
        u4[j] = du4_d[j];
        du4[j] = u4[j] - u4_mean_d[j];
      }
    }
  }
  else if (control_dim % 2 == 0)
  {
    float2* du2 = reinterpret_cast<float2*>(du_thread);
    float2* u2 = reinterpret_cast<float2*>(u_thread);
    const float2* u2_mean_d = reinterpret_cast<const float2*>(&u_d[t * control_dim]);
    const float2* du2_d = reinterpret_cast<const float2*>(&du_d[control_index]);
    for (int j = thread_idy; j < control_dim / 2; j += blocksize_y)
    {
      if (global_idx == 0)
      {
        du2[j] = make_float2(0, 0);
        u2[j] = u2_mean_d[j];
      }
      else if (global_idx >= 0.99 * num_rollouts)
      {
        du2[j] = du2_d[j];
        u2[j] = du2[j];
      }
      else
      {
        u2[j] = du2_d[j];
        du2[j] = u2[j] - u2_mean_d[j];
      }
    }
  }
  else
  {
    for (int j = thread_idy; j < control_dim; j += blockDim.y)
    {
      if (global_idx == 0)
      {
        du_thread[j] = 0;
        u_thread[j] = u_d[t * control_dim + j];
      }
      else if (global_idx >= 0.99 * num_rollouts)
      {
        du_thread[j] = du_d[control_index + j];
        u_thread[j] = du_thread[j];
      }
      else
      {
        u_thread[j] = du_d[control_index + j];
        du_thread[j] = u_thread[j] - u_d[t * control_dim + j];
      }
    }
  }
}

// TODO generalize the trim control
// The zero control trajectory should be an equilbrium control defined in the dynamics.
__device__ void injectControlNoise(int control_dim, int blocksize_y, int num_rollouts, int num_timesteps,
                                   int current_timestep, int global_idx, int thread_idy, int optimization_stride,
                                   const float* u_traj_device, float* ep_v_device, const float* sigma_u_thread,
                                   float* u_thread, float* du_thread)
{
  // this is a global index
  int control_index = (num_rollouts * num_timesteps * threadIdx.z +  // z part
                       global_idx * num_timesteps + current_timestep) *
                      control_dim;  // normal part
  // Load the noise trajectory scaled by the exploration factor
  // The prior loop already guarantees that the global index is less than the number of rollouts
  if (control_dim % 4 == 0)
  {
    float4* u4_thread = reinterpret_cast<float4*>(u_thread);
    float4* du4_thread = reinterpret_cast<float4*>(du_thread);
    const float4* sigma_u4_thread = reinterpret_cast<const float4*>(sigma_u_thread);
    const float4* u4_traj_device = reinterpret_cast<const float4*>(&u_traj_device[current_timestep * control_dim]);
    float4* ep4_v_device = reinterpret_cast<float4*>(&ep_v_device[control_index]);
    for (int i = thread_idy; i < control_dim / 4; i += blocksize_y)
    {
      // Keep one noise free trajectory
      if (global_idx == 0 || current_timestep < optimization_stride)
      {
        du4_thread[i] = make_float4(0, 0, 0, 0);
        u4_thread[i] = u4_traj_device[i];
      }
      // Generate 1% zero control trajectory
      else if (global_idx >= 0.99 * num_rollouts)
      {
        du4_thread[i] = ep4_v_device[i] * sigma_u4_thread[i];
        u4_thread[i] = du4_thread[i];
      }
      else
      {
        du4_thread[i] = ep4_v_device[i] * sigma_u4_thread[i];
        u4_thread[i] = u4_traj_device[i] + du4_thread[i];
      }
      // Saves the control but doesn't clamp it.
      ep4_v_device[i] = u4_thread[i];
    }
  }
  else if (control_dim % 2 == 0)
  {
    float2* u2_thread = reinterpret_cast<float2*>(u_thread);
    float2* du2_thread = reinterpret_cast<float2*>(du_thread);
    const float2* sigma_u2_thread = reinterpret_cast<const float2*>(sigma_u_thread);
    const float2* u2_traj_device = reinterpret_cast<const float2*>(&u_traj_device[current_timestep * control_dim]);
    float2* ep2_v_device = reinterpret_cast<float2*>(&ep_v_device[control_index]);
    for (int i = thread_idy; i < control_dim / 2; i += blocksize_y)
    {
      // Keep one noise free trajectory
      if (global_idx == 0 || current_timestep < optimization_stride)
      {
        du2_thread[i] = make_float2(0, 0);
        u2_thread[i] = u2_traj_device[i];
      }
      // Generate 1% zero control trajectory
      else if (global_idx >= 0.99 * num_rollouts)
      {
        du2_thread[i] = ep2_v_device[i] * sigma_u2_thread[i];
        u2_thread[i] = du2_thread[i];
      }
      else
      {
        du2_thread[i] = ep2_v_device[i] * sigma_u2_thread[i];
        u2_thread[i] = u2_traj_device[i] + du2_thread[i];
      }
      // Saves the control but doesn't clamp it.
      ep2_v_device[i] = u2_thread[i];
    }
  }
  else
  {
    for (int i = thread_idy; i < control_dim; i += blocksize_y)
    {
      // Keep one noise free trajectory
      if (global_idx == 0 || current_timestep < optimization_stride)
      {
        du_thread[i] = 0;
        u_thread[i] = u_traj_device[current_timestep * control_dim + i];
      }
      // Generate 1% zero control trajectory
      else if (global_idx >= 0.99 * num_rollouts)
      {
        du_thread[i] = ep_v_device[control_index + i] * sigma_u_thread[i];
        u_thread[i] = du_thread[i];
      }
      else
      {
        du_thread[i] = ep_v_device[control_index + i] * sigma_u_thread[i];
        u_thread[i] = u_traj_device[current_timestep * control_dim + i] + du_thread[i];
      }
      // Saves the control but doesn't clamp it.
      ep_v_device[control_index + i] = u_thread[i];
    }
  }
}

template <class COST_T>
__device__ void computeAndSaveCost(int num_rollouts, int num_timesteps, int global_idx, COST_T* costs, float* output,
                                   float running_cost, float* theta_c, float* cost_rollouts_device)
{
  // only want to save 1 cost per trajectory
  if (threadIdx.y == 0 && global_idx < num_rollouts)
  {
    cost_rollouts_device[global_idx + num_rollouts * threadIdx.z] =
        running_cost + costs->terminalCost(output, theta_c) / (num_timesteps - 1);
  }
}

/*******************************************************************************************************************
 * NormExp Kernel Helpers
 *******************************************************************************************************************/
float computeBaselineCost(float* cost_rollouts_host, int num_rollouts)
{  // TODO if we use standard containers in MPPI, should this be replaced with a min algorithm?
  int best_idx = computeBestIndex(cost_rollouts_host, num_rollouts);
  return cost_rollouts_host[best_idx];
}

float constructBestWeights(float* cost_rollouts_host, int num_rollouts)
{
  int best_idx = computeBestIndex(cost_rollouts_host, num_rollouts);
  float best_cost = cost_rollouts_host[best_idx];

  for (int i = 0; i < num_rollouts; i++)
  {
    if (i == best_idx)
    {
      cost_rollouts_host[i] = 1.0;
    }
    else
    {
      cost_rollouts_host[i] = 0.0;
    }
  }

  // printf("Best idx: %d, cost: %f\n", best_cost_idx, best_cost);
  return best_cost;
}

int computeBestIndex(float* cost_rollouts_host, int num_rollouts)
{
  float best_cost = cost_rollouts_host[0];
  int best_cost_idx = 0;
  for (int i = 1; i < num_rollouts; i++)
  {
    if (cost_rollouts_host[i] < best_cost)
    {
      best_cost = cost_rollouts_host[i];
      best_cost_idx = i;
    }
  }

  // printf("Best idx: %d, cost: %f\n", best_cost_idx, best_cost);
  return best_cost_idx;
}

__device__ inline float computeBaselineCost(int num_rollouts, const float* __restrict__ trajectory_costs_d,
                                            float* __restrict__ reduction_buffer, int rollout_idx_global,
                                            int rollout_idx_step)
{
  // Copy costs to shared memory
  float min_cost = 0.0;
#if false
  // potential method to speed up copying costs
  int prev_size = min(blockDim.x, num_rollouts);
  float my_val = (rollout_idx_global < num_rollouts) ? trajectory_costs_d[rollout_idx_global] : INFINITY;
  for (int i = rollout_idx_global + rollout_idx_step; i < num_rollouts; i += rollout_idx_step)
  {
    my_val = min(trajectory_costs_d[i], my_val);
  }
  reduction_buffer[rollout_idx_global] = my_val;
  // __syncthreads();
  // if (threadIdx.x == 0)
  // {
  //   for (int i = 0; i < min(blockDim.x, num_rollouts); i++)
  //   {
  //     printf("buff %d: %f\n", i, reduction_buffer[i]);
  //   }
  //   printf("Num rollouts; %d\n", num_rollouts);
  // }
#else
  int prev_size = num_rollouts / 2;
  for (int i = rollout_idx_global; i < prev_size; i += rollout_idx_step)
  {
    reduction_buffer[i] = min(trajectory_costs_d[i], trajectory_costs_d[i + prev_size]);
  }
  if (num_rollouts - 2 * prev_size == 1 && threadIdx.x == blockDim.x - 1)
  {
    reduction_buffer[prev_size - 1] = min(reduction_buffer[num_rollouts - 1], reduction_buffer[prev_size - 1]);
  }
#endif

  __syncthreads();
  // find min along the entire array
  for (int size = prev_size / 2; size > 0; size /= 2)
  {
    for (int i = rollout_idx_global; i < size; i += rollout_idx_step)
    {
      reduction_buffer[i] = min(reduction_buffer[i], reduction_buffer[i + size]);
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1)
    {
      reduction_buffer[size - 1] = min(reduction_buffer[size - 1], reduction_buffer[prev_size - 1]);
    }
    __syncthreads();
    prev_size = size;
  }
  min_cost = reduction_buffer[0];
  return min_cost;
}

__device__ __host__ inline void normExpTransform(int num_rollouts, float* __restrict__ trajectory_costs_d,
                                                 float lambda_inv, float baseline, int global_idx, int rollout_idx_step)
{
  for (int i = global_idx; i < num_rollouts; i += rollout_idx_step)
  {
    float cost_dif = trajectory_costs_d[i] - baseline;
    trajectory_costs_d[i] = expf(-lambda_inv * cost_dif);
  }
}

__device__ __host__ inline void TsallisTransform(int num_rollouts, float* __restrict__ trajectory_costs_d, float gamma,
                                                 float r, float baseline, int global_idx, int rollout_idx_step)
{
  for (int i = global_idx; i < num_rollouts; i += rollout_idx_step)
  {
    float cost_dif = trajectory_costs_d[i] - baseline;
    // trajectory_costs_d[i] = mppi::math::expr(-lambda_bar_inv * cost_dif);
    // trajectory_costs_d[i] = (cost_dif < gamma) * expf(logf(1.0 - cost_dif / gamma) / (r - 1));
    if (cost_dif < gamma)
    {
      trajectory_costs_d[i] = expf(logf(1.0 - cost_dif / gamma) / (r - 1));
    }
    else
    {
      trajectory_costs_d[i] = 0;
    }
  }
}

__device__ inline float computeNormalizer(int num_rollouts, const float* __restrict__ trajectory_costs_d,
                                          float* __restrict__ reduction_buffer, int rollout_idx_global,
                                          int rollout_idx_step)
{
  // Copy costs to shared memory
#if false
  // potential method to speed up copying costs
  int prev_size = min(blockDim.x, num_rollouts);
  float my_val = (rollout_idx_global < num_rollouts) ? trajectory_costs_d[rollout_idx_global] : 0;
  for (int i = rollout_idx_global + rollout_idx_step; i < num_rollouts; i += rollout_idx_step)
  {
    my_val += trajectory_costs_d[i];
  }
  reduction_buffer[rollout_idx_global] = my_val;
#else
  int prev_size = num_rollouts / 2;
  for (int i = rollout_idx_global; i < prev_size; i += rollout_idx_step)
  {
    reduction_buffer[i] = trajectory_costs_d[i] + trajectory_costs_d[i + prev_size];
  }
  if (num_rollouts - 2 * prev_size == 1 && threadIdx.x == blockDim.x - 1)
  {
    reduction_buffer[prev_size - 1] += reduction_buffer[num_rollouts - 1];
  }
#endif
  __syncthreads();
  // sum the entire array
  for (int size = prev_size / 2; size > 0; size /= 2)
  {
    for (int i = rollout_idx_global; i < size; i += rollout_idx_step)
    {
      reduction_buffer[i] += reduction_buffer[i + size];
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1)
    {
      reduction_buffer[size - 1] += reduction_buffer[prev_size - 1];
    }
    __syncthreads();
    prev_size = size;
  }
  return reduction_buffer[0];
}

template <int NUM_ROLLOUTS, int BLOCKSIZE_X = 1024>
__global__ void fullGPUcomputeWeights(float* __restrict__ trajectory_costs_d, float lambda_inv,
                                      float2* __restrict__ output)
{
  __shared__ float reduction_buffer[NUM_ROLLOUTS];
  // int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // int better_global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * blockDim.y + threadIdx.y;
  // int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * blockDim.z + threadIdx.z;
  // int global_step = blockDim.x * gridDim.x;
  // int better_global_step = blockDim.x * gridDim.x  * blockDim.y * gridDim.y;
  int global_idx = threadIdx.x;
  int global_step = blockDim.x;

  float baseline = computeBaselineCost(NUM_ROLLOUTS, trajectory_costs_d, reduction_buffer, global_idx, global_step);
  normExpTransform(NUM_ROLLOUTS, trajectory_costs_d, lambda_inv, baseline, global_idx, global_step);
  __syncthreads();
  float normalizer = computeNormalizer(NUM_ROLLOUTS, trajectory_costs_d, reduction_buffer, global_idx, global_step);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    *output = make_float2(baseline, normalizer);
  }
}

float computeNormalizer(float* cost_rollouts_host, int num_rollouts)
{
  double normalizer = 0.0;
  for (int i = 0; i < num_rollouts; ++i)
  {
    normalizer += cost_rollouts_host[i];
  }
  return normalizer;
}

void computeFreeEnergy(float& free_energy, float& free_energy_var, float& free_energy_modified,
                       float* cost_rollouts_host, int num_rollouts, float baseline, float lambda)
{
  float var = 0;
  float norm = 0;
  for (int i = 0; i < num_rollouts; i++)
  {
    norm += cost_rollouts_host[i];
    var += powf(cost_rollouts_host[i], 2);
  }
  norm /= num_rollouts;
  free_energy = -lambda * logf(norm) + baseline;
  free_energy_var = lambda * (var / num_rollouts - powf(norm, 2));
  // TODO Figure out the point of the following lines
  float weird_term = free_energy_var / (norm * sqrtf(1.0 * num_rollouts));
  free_energy_modified = lambda * (weird_term + 0.5 * powf(weird_term, 2));
}

/*******************************************************************************************************************
 * Weighted Reduction Kernel Helpers
 *******************************************************************************************************************/
__device__ void setInitialControlToZero(int control_dim, int thread_idx, float* u, float* u_intermediate)
{
  for (int i = 0; i < control_dim; i++)
  {
    u[i] = 0;
    u_intermediate[thread_idx * control_dim + i] = 0;
  }
}

__device__ void strideControlWeightReduction(int num_rollouts, int num_timesteps, int sum_stride, int thread_idx,
                                             int block_idx, int control_dim, float* exp_costs_d, float normalizer,
                                             float* du_d, float* u, float* u_intermediate)
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

__device__ void rolloutWeightReductionAndSaveControl(int thread_idx, int block_idx, int num_rollouts, int num_timesteps,
                                                     int control_dim, int sum_stride, float* u, float* u_intermediate,
                                                     float* du_new_d)
{
  if (thread_idx == 0 && block_idx < num_timesteps)
  {  // block index refers to the current timestep
    for (int i = 0; i < control_dim; ++i)
    {  // TODO replace with memset?
      u[i] = 0;
    }
    for (int i = 0; i < ((num_rollouts - 1) / sum_stride + 1); ++i)
    {  // iterate through the each subsection
      for (int j = 0; j < control_dim; ++j)
      {
        u[j] += u_intermediate[i * control_dim + j];
      }
    }
    for (int i = 0; i < control_dim; i++)
    {
      du_new_d[block_idx * control_dim + i] = u[i];
    }
  }
}

template <class COST_T, int BLOCKSIZE_X, bool COALESCE = false>
__global__ void visualizeCostKernel(COST_T* costs, float dt, const int num_timesteps, const int num_rollouts,
                                    const float lambda, float alpha, const float* __restrict__ u_d,
                                    const float* __restrict__ du_d, const float* __restrict__ sigma_u_d,
                                    const float* __restrict__ y_d, float* __restrict__ cost_traj_d,
                                    int* __restrict__ crash_status_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];
  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[blockDim.x * blockDim.z * COST_T::OUTPUT_DIM];
  float* du_shared = &u_shared[blockDim.x * blockDim.z * COST_T::CONTROL_DIM];
  float* sigma_u = &du_shared[blockDim.x * blockDim.z * COST_T::CONTROL_DIM];
  float* running_cost_shared = &sigma_u[COST_T::CONTROL_DIM];
  int* crash_status_shared = (int*)&running_cost_shared[blockDim.z * num_timesteps];
  float* theta_c = (float*)&crash_status_shared[blockDim.z * blockDim.x];

  // Create local state, state dot and controls
  float* y;
  float* u;
  float* du;
  int* crash_status;

  // Initialize running cost and total cost
  float* running_cost;
  int sample_time_offset = 0;
  int cost_index = 0;

  // Load global array to shared array
  y = &y_shared[(blockDim.x * thread_idz + thread_idx) * COST_T::OUTPUT_DIM];
  u = &u_shared[(blockDim.x * thread_idz + thread_idx) * COST_T::CONTROL_DIM];
  du = &du_shared[(blockDim.x * thread_idz + thread_idx) * COST_T::CONTROL_DIM];
  crash_status = &crash_status_shared[thread_idz * blockDim.x + thread_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  // running_cost = &running_cost_shared[thread_idz * blockDim.x + thread_idx];
  // running_cost[0] = 0;
  if (thread_idx == 0)
  {
    mp1::loadArrayParallel<COST_T::CONTROL_DIM>(sigma_u, 0, sigma_u_d, 0);
  }

  /*<----Start of simulation loop-----> */
  const int max_time_iters = ceilf((float)num_timesteps / BLOCKSIZE_X);
  costs->initializeCosts(y, u, theta_c, 0.0, dt);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x + 1;
    cost_index = (thread_idz * num_rollouts + global_idx) * (num_timesteps) + t - 1;
    running_cost = &running_cost_shared[thread_idz * blockDim.x + t - 1];
    if (t <= num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      if (COALESCE)
      {  // Fill entire shared mem sequentially using sequential threads_idx
        mp1::loadArrayParallel<COST_T::OUTPUT_DIM * BLOCKSIZE_X, mp1::Parallel1Dir::THREAD_X>(
            y_shared, blockDim.x * thread_idz, y_d,
            ((num_rollouts * thread_idz + global_idx) * num_timesteps + time_iter * blockDim.x) * COST_T::OUTPUT_DIM);
      }
      else
      {
        sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t - 1;
        mp1::loadArrayParallel<COST_T::OUTPUT_DIM>(y, 0, y_d, sample_time_offset * COST_T::OUTPUT_DIM);
      }
    }
    if (t < num_timesteps)
    {  // load controls from t = 0 to t = num_timesteps - 1
      // Have to do similar steps as injectControlNoise but using the already transformed cost samples
      readControlsFromGlobal(COST_T::CONTROL_DIM, blockDim.y, num_rollouts, num_timesteps, t, global_idx, thread_idy,
                             u_d, du_d, u, du);
    }
    __syncthreads();

    // Compute cost
    if (thread_idy == 0 && t < num_timesteps)
    {
      // cost_traj_d[cost_index] =
      running_cost[0] =
          costs->computeRunningCost(y, u, du, sigma_u, lambda, alpha, t, theta_c, crash_status) / (num_timesteps - 1);
      crash_status_d[global_idx * num_timesteps + t] = crash_status[0];
    }
    __syncthreads();
  }
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % BLOCKSIZE_X;
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
  if (((unsigned long)running_cost_shared & 15) == 0 && (num_timesteps) % 4 == 0)
  {
    for (int i = thread_idx; i < num_timesteps / 4; i += blockDim.x)
    {
      reinterpret_cast<float4*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps])[i] =
          reinterpret_cast<float4*>(&running_cost_shared[thread_idz * num_timesteps])[i];
    }
  }
  else if (((unsigned long)running_cost_shared & 7) == 0 && (num_timesteps) % 2 == 0)
  {
    for (int i = thread_idx; i < num_timesteps / 2; i += blockDim.x)
    {
      reinterpret_cast<float2*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps])[i] =
          reinterpret_cast<float2*>(&running_cost_shared[thread_idz * num_timesteps])[i];
    }
  }
  else
  {
    for (int i = thread_idx; i < num_timesteps; i += blockDim.x)
    {
      cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps + i] =
          running_cost_shared[thread_idz * num_timesteps + i];
    }
  }
}

template <class DYN_T, class COST_T, class FB_T, int BLOCKSIZE_X, int BLOCKSIZE_Z>
__global__ void stateAndCostTrajectoryKernel(DYN_T* dynamics, COST_T* costs, FB_T* fb_controller, float* control,
                                             float* state, float* output_traj_d, float* cost_traj_d,
                                             int* crash_status_d, int num_rollouts, int num_timesteps, float dt,
                                             float value_func_threshold)
{
  // Get thread and block id
  int thread_idx = threadIdx.x;
  int thread_idy = threadIdx.y;
  int thread_idz = threadIdx.z;
  int block_idx = blockIdx.x;
  int global_idx = BLOCKSIZE_X * block_idx + thread_idx;

  // Create shared state and control arrays
  __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float x_next_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float y_shared[BLOCKSIZE_X * DYN_T::OUTPUT_DIM * BLOCKSIZE_Z];
  __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];

  // Create a shared array for the nominal costs calculations
  __shared__ int crash_status_shared[BLOCKSIZE_X * BLOCKSIZE_Z];

  // Create a shared array for the dynamics model to use
  __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 +
                           DYN_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X * BLOCKSIZE_Z];
  __shared__ float
      theta_c[COST_T::SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 + COST_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X];
  __shared__ float theta_fb[FB_T::SHARED_MEM_SIZE];

  // Create local state, state dot and controls
  float* x;
  float* x_next;
  float* x_temp;
  float* y;
  float* x_other;
  float* xdot;
  float* u;
  int* crash_status;
  float fb_control[DYN_T::CONTROL_DIM];
  int t_index = 0;
  int cost_index = 0;

  if (global_idx < num_rollouts)
  {
    // Actual or nominal
    x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    x_next = &x_next_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    y = &y_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::OUTPUT_DIM];
    // The opposite state from above
    x_other = &x_shared[(blockDim.x * (1 - thread_idz) + thread_idx) * DYN_T::STATE_DIM];
    xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    // Base trajectory
    u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    // Nominal State Cost
    crash_status = &crash_status_shared[thread_idz * blockDim.x + thread_idx];
    crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.

    // Load memory into appropriate arrays
    for (int i = thread_idy; i < DYN_T::STATE_DIM; i += blockDim.y)
    {
      x[i] = state[DYN_T::STATE_DIM * threadIdx.z + i];
      xdot[i] = 0.0;
    }
    __syncthreads();
    float curr_state_cost = 0.0;

    dynamics->initializeDynamics(x, u, y, theta_s, 0.0, dt);
    costs->initializeCosts(y, u, theta_c, 0.0, dt);
    for (int t = 0; t < num_timesteps; t++)
    {
      t_index = threadIdx.z * num_rollouts * num_timesteps + global_idx * num_timesteps + t;
      cost_index = threadIdx.z * num_rollouts * (num_timesteps + 1) + global_idx * (num_timesteps + 1) + t;
      // get next u
      for (int i = thread_idy; i < DYN_T::CONTROL_DIM; i += blockDim.y)
      {
        u[i] = control[global_idx * num_timesteps * DYN_T::CONTROL_DIM + t * DYN_T::CONTROL_DIM + i];
      }

      // only apply feedback if enabled
      // feedback is only applied on real state in RMPPI
      if (BLOCKSIZE_Z > 1 && value_func_threshold == -1 && thread_idz == 0)
      {
        fb_controller->k(x, x_other, t, theta_fb, fb_control);

        for (int i = thread_idy; i < DYN_T::CONTROL_DIM; i += blockDim.y)
        {
          u[i] += fb_control[i];
        }
      }
      __syncthreads();

      dynamics->enforceConstraints(x, u);
      __syncthreads();

      if (thread_idy == 0 && t > 0)
      {
        curr_state_cost = costs->computeStateCost(y, t, theta_c, crash_status);
        crash_status_d[t_index] = crash_status[0];
        cost_traj_d[cost_index] = curr_state_cost;
      }
      __syncthreads();
      // Nominal system is where thread_idz == 1
      if (thread_idz == 1 && thread_idy == 0 && t > 0)
      {
        // compute the nominal system cost
        cost_traj_d[cost_index] =
            0.5 * curr_state_cost +
            // here we know threadIdx.z == 0 since we are only talking about the real system
            fmaxf(fminf(cost_traj_d[global_idx * (num_timesteps + 1) + t], value_func_threshold), curr_state_cost);
      }
      __syncthreads();
      // reset crash status in case initial location is actually a crash cost
      if (t == 0)
      {
        crash_status[0] = 0;
      }

      // Increment states
      dynamics->step(x, x_next, xdot, u, y, theta_s, t, dt);
      __syncthreads();
      x_temp = x;
      x = x_next;
      x_next = x_temp;

      // save results, skips the first state location since that is known
      for (int i = thread_idy; i < DYN_T::OUTPUT_DIM; i += blockDim.y)
      {
        output_traj_d[t_index * DYN_T::OUTPUT_DIM + i] = y[i];
      }
    }
    // get cost traj at +1
    cost_index = threadIdx.z * num_rollouts * (num_timesteps + 1) + global_idx * (num_timesteps + 1) + num_timesteps;
    cost_traj_d[cost_index] = costs->terminalCost(y, theta_c) / (num_timesteps - 1);
  }
}

/*******************************************************************************************************************
 * Launch Functions
 *******************************************************************************************************************/
template <class DYN_T, class COST_T, int NUM_ROLLOUTS, int BLOCKSIZE_X, int BLOCKSIZE_Y, int BLOCKSIZE_Z>
void launchRolloutKernel(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, int optimization_stride,
                         float lambda, float alpha, float* x_d, float* u_d, float* du_d, float* sigma_u_d,
                         float* trajectory_costs, cudaStream_t stream, bool synchronize)
{
  const int gridsize_x = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;
  static_assert(NUM_ROLLOUTS % BLOCKSIZE_X == 0, "NUM_ROLLOUTS must be evenly divided by BLOCKSIZE_X");
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  dim3 dimGrid(gridsize_x, 1, 1);
  rolloutKernel<DYN_T, COST_T, BLOCKSIZE_X, BLOCKSIZE_Y, NUM_ROLLOUTS, BLOCKSIZE_Z>
      <<<dimGrid, dimBlock, 0, stream>>>(dynamics, costs, dt, num_timesteps, optimization_stride, lambda, alpha, x_d,
                                         u_d, du_d, sigma_u_d, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, class COST_T, int NUM_ROLLOUTS, int DYN_BLOCK_X, int DYN_BLOCK_Y, int BLOCKSIZE_Z,
          int COST_BLOCK_X, int COST_BLOCK_Y>
void launchFastRolloutKernel(DYN_T* dynamics, COST_T* costs, float dt, const int num_timesteps, int optimization_stride,
                             float lambda, float alpha, float* init_x_d, float* x_d, float* u_d, float* du_d,
                             float* sigma_u_d, float* trajectory_costs, cudaStream_t stream, bool synchronize)
{
  // Run Dynamics
  const int gridsize_x = (NUM_ROLLOUTS - 1) / DYN_BLOCK_X + 1;
  dim3 dimBlock(DYN_BLOCK_X, DYN_BLOCK_Y, BLOCKSIZE_Z);
  dim3 dimGrid(gridsize_x, 1, 1);
  rolloutDynamicsKernel<DYN_T, DYN_BLOCK_X, DYN_BLOCK_Y, NUM_ROLLOUTS, BLOCKSIZE_Z><<<dimGrid, dimBlock, 0, stream>>>(
      dynamics, dt, num_timesteps, optimization_stride, init_x_d, u_d, du_d, sigma_u_d, x_d);

  // Run Costs
  dim3 dimCostBlock(COST_BLOCK_X, COST_BLOCK_Y, BLOCKSIZE_Z);
  dim3 dimCostGrid(NUM_ROLLOUTS, 1, 1);
  unsigned shared_mem_size =
      ((COST_BLOCK_X * BLOCKSIZE_Z) * (DYN_T::OUTPUT_DIM + 2 * DYN_T::CONTROL_DIM + 1) + DYN_T::CONTROL_DIM) *
          sizeof(float) +
      (COST_BLOCK_X * BLOCKSIZE_Z) * sizeof(int) + COST_T::SHARED_MEM_REQUEST_GRD +
      COST_T::SHARED_MEM_REQUEST_BLK * COST_BLOCK_X * BLOCKSIZE_Z * sizeof(float);
  rolloutCostKernel<DYN_T, COST_T, NUM_ROLLOUTS, COST_BLOCK_X><<<dimCostGrid, dimCostBlock, shared_mem_size, stream>>>(
      dynamics, costs, dt, num_timesteps, lambda, alpha, init_x_d, u_d, du_d, sigma_u_d, x_d, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

void launchNormExpKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float lambda_inv, float baseline,
                         cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock(blocksize_x, 1, 1);
  dim3 dimGrid((num_rollouts - 1) / blocksize_x + 1, 1, 1);
  normExpKernel<<<dimGrid, dimBlock, 0, stream>>>(num_rollouts, trajectory_costs_d, lambda_inv, baseline);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

void launchTsallisKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float gamma, float r,
                         float baseline, cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock(blocksize_x, 1, 1);
  dim3 dimGrid((num_rollouts - 1) / blocksize_x + 1, 1, 1);
  TsallisKernel<<<dimGrid, dimBlock, 0, stream>>>(num_rollouts, trajectory_costs_d, gamma, r, baseline);
  // CudaCheckError();
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <int NUM_ROLLOUTS>
void launchWeightTransformKernel(float* __restrict__ costs_d, float2* __restrict__ baseline_and_norm_d,
                                 const float lambda_inv, const int num_systems, cudaStream_t stream, bool synchronize)
{
  // Figure out max size of threads from the device properties (slows down this method a lot)
  // int device_id = 0;
  // cudaDeviceProp deviceProp;
  // cudaGetDeviceProperties(&deviceProp, device_id);
  // int blocksize_x = deviceProp.maxThreadsDim[0];
  const int blocksize_x = 1024;
  dim3 dimBlock(blocksize_x, 1, 1);
  // Can't be split into multiple blocks because we want to do all the math in shared memory
  dim3 dimGrid(1, 1, 1);
  for (int i = 0; i < num_systems; i++)
  {
    fullGPUcomputeWeights<NUM_ROLLOUTS>
        <<<dimGrid, dimBlock, 0, stream>>>(costs_d + i * NUM_ROLLOUTS, lambda_inv, baseline_and_norm_d + i);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, int NUM_ROLLOUTS, int SUM_STRIDE>
void launchWeightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d, float normalizer,
                                   int num_timesteps, cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1, 1, 1);
  dim3 dimGrid(num_timesteps, 1, 1);
  weightedReductionKernel<DYN_T::CONTROL_DIM, NUM_ROLLOUTS, SUM_STRIDE>
      <<<dimGrid, dimBlock, 0, stream>>>(exp_costs_d, du_d, du_new_d, normalizer, num_timesteps);
  // CudaCheckError();
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, int NUM_ROLLOUTS, int SUM_STRIDE>
void launchweightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d, float2* baseline_and_normalizer_d,
                                   int num_timesteps, cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1, 1, 1);
  dim3 dimGrid(num_timesteps, 1, 1);
  weightedReductionKernel<DYN_T::CONTROL_DIM, NUM_ROLLOUTS, SUM_STRIDE>
      <<<dimGrid, dimBlock, 0, stream>>>(exp_costs_d, du_d, du_new_d, baseline_and_normalizer_d, num_timesteps);
  // CudaCheckError();
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, class COST_T, class FB_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int BLOCKSIZE_Z>
void launchStateAndCostTrajectoryKernel(DYN_T* dynamics, COST_T* cost, FB_T* fb_controller, float* control_trajectories,
                                        float* state, float* output_traj_result, float* cost_traj_result,
                                        int* crash_status_result, int num_rollouts, int num_timesteps, float dt,
                                        cudaStream_t stream, float value_func_threshold, bool synchronize)
{
  const int gridsize_x = (num_rollouts - 1) / BLOCKSIZE_X + 1;
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  dim3 dimGrid(gridsize_x, 1, 1);
  stateAndCostTrajectoryKernel<DYN_T, COST_T, FB_T, BLOCKSIZE_X, BLOCKSIZE_Z><<<dimGrid, dimBlock, 0, stream>>>(
      dynamics, cost, fb_controller, control_trajectories, state, output_traj_result, cost_traj_result,
      crash_status_result, num_rollouts, num_timesteps, dt, value_func_threshold);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class COST_T, int COST_BLOCK_X, int COST_BLOCK_Y, int BLOCKSIZE_Z>
void launchVisualizeCostKernel(COST_T* costs, float dt, const int num_timesteps, const int num_rollouts, float lambda,
                               float alpha, float* y_d, float* sampled_controls_d, int* sampled_crash_status_d,
                               float* sigma_u_d, float* cost_traj_result, cudaStream_t stream, bool synchronize)
{
  if (num_rollouts <= 1)
  {  // Not enough samples to visualize
    std::cerr << "Not enough samples to visualize" << std::endl;
    return;
  }
  // Run Costs
  dim3 dimCostBlock(COST_BLOCK_X, COST_BLOCK_Y, BLOCKSIZE_Z);
  dim3 dimCostGrid(num_rollouts, 1, 1);
  unsigned shared_mem_size =
      ((COST_BLOCK_X * BLOCKSIZE_Z) * (COST_T::OUTPUT_DIM + 2 * COST_T::CONTROL_DIM) + COST_T::CONTROL_DIM) *
          sizeof(float) +
      sizeof(float) * (num_timesteps * BLOCKSIZE_Z) + (COST_BLOCK_X * BLOCKSIZE_Z) * sizeof(int) +
      COST_T::SHARED_MEM_REQUEST_GRD + COST_T::SHARED_MEM_REQUEST_BLK * COST_BLOCK_X * BLOCKSIZE_Z * sizeof(float);
  visualizeCostKernel<COST_T, COST_BLOCK_X><<<dimCostGrid, dimCostBlock, shared_mem_size, stream>>>(
      costs, dt, num_timesteps, num_rollouts - 1, lambda, alpha, sampled_controls_d,
      sampled_controls_d + COST_T::CONTROL_DIM * num_timesteps, sigma_u_d, y_d, cost_traj_result,
      sampled_crash_status_d);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}
}  // namespace mppi_common

namespace rmppi_kernels
{
template <class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int SAMPLES_PER_CONDITION>
__global__ void initEvalKernel(DYN_T* dynamics, COST_T* costs, int num_timesteps, float lambda, float alpha,
                               int ctrl_stride, float dt, int* strides_d, float* exploration_std_dev_d, float* states_d,
                               float* control_d, float* control_noise_d, float* costs_d)
{
  int i, j;
  int tdx = threadIdx.x;
  int tdy = threadIdx.y;
  int bdx = blockIdx.x;

  // Initialize the local state, controls, and noise
  float* state;
  float* state_next;
  float* state_temp;
  float* output;
  float* state_der;
  float* control;
  float* control_noise;  // du
  int* crash_status;

  // Create shared arrays for holding state and control data.
  __shared__ float state_shared[BLOCKSIZE_X * DYN_T::STATE_DIM];
  __shared__ float state_next_shared[BLOCKSIZE_X * DYN_T::STATE_DIM];
  __shared__ float output_shared[BLOCKSIZE_X * DYN_T::OUTPUT_DIM];
  __shared__ float state_der_shared[BLOCKSIZE_X * DYN_T::STATE_DIM];
  __shared__ float control_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM];
  __shared__ float control_noise_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM];
  __shared__ float exploration_std_dev[DYN_T::CONTROL_DIM];  // Each thread only reads
  __shared__ int crash_status_shared[BLOCKSIZE_X];

  // Create a shared array for the dynamics model to use
  __shared__ float
      theta_s[DYN_T::SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 + DYN_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X];
  __shared__ float theta_c[COST_T::SHARED_MEM_REQUEST_GRD + COST_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X];

  float running_cost = 0;  // Initialize trajectory cost

  int global_idx = BLOCKSIZE_X * bdx + tdx;                // Set the global index for CUDA threads
  int condition_idx = global_idx / SAMPLES_PER_CONDITION;  // Set the index for our candidate
  int stride = strides_d[condition_idx];                   // Each candidate can have a different starting stride

  // Get the pointer that belongs to the current thread with respect to the shared arrays
  state = &state_shared[tdx * DYN_T::STATE_DIM];
  state_next = &state_next_shared[tdx * DYN_T::STATE_DIM];
  output = &output_shared[tdx * DYN_T::OUTPUT_DIM];
  state_der = &state_der_shared[tdx * DYN_T::STATE_DIM];
  control = &control_shared[tdx * DYN_T::CONTROL_DIM];
  control_noise = &control_noise_shared[tdx * DYN_T::CONTROL_DIM];
  crash_status = &crash_status_shared[tdx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.

  // Copy the state to the thread
  for (i = tdy; i < DYN_T::STATE_DIM; i += blockDim.y)
  {
    state[i] = states_d[condition_idx * DYN_T::STATE_DIM + i];  // states_d holds each condition
  }

  // Copy the exploration noise std_dev to the thread
  for (i = tdy; i < DYN_T::CONTROL_DIM; i += blockDim.y)
  {
    control[i] = 0.0;
    control_noise[i] = 0.0;
    exploration_std_dev[i] = exploration_std_dev_d[i];
  }

  __syncthreads();
  dynamics->initializeDynamics(state, control, output, theta_s, 0.0, dt);
  costs->initializeCosts(output, control, theta_c, 0.0, dt);
  for (i = 0; i < num_timesteps; ++i)
  {  // Outer loop iterates on timesteps
    // Inject the control noise
    for (j = tdy; j < DYN_T::CONTROL_DIM; j += blockDim.y)
    {
      if ((i + stride) >= num_timesteps)
      {  // Pad the end of the controls with the last control
        control[j] = control_d[(num_timesteps - 1) * DYN_T::CONTROL_DIM + j];
      }
      else
      {
        control[j] = control_d[(i + stride) * DYN_T::CONTROL_DIM + j];
      }

      // First rollout is noise free
      if (global_idx % SAMPLES_PER_CONDITION == 0 || i < ctrl_stride)
      {
        control_noise[j] = 0.0;
      }
      else
      {
        control_noise[j] =
            control_noise_d[num_timesteps * DYN_T::CONTROL_DIM * global_idx + i * DYN_T::CONTROL_DIM + j] *
            exploration_std_dev[j];
      }

      // Sum the control and the noise
      control[j] += control_noise[j];
    }  // End inject control noise

    __syncthreads();

    dynamics->enforceConstraints(state, control);
    __syncthreads();
    if (tdy == 0 && i > 0)
    {  // Only compute once per global index, make sure that we don't divide by zero
      running_cost += (costs->computeRunningCost(output, control, control_noise, exploration_std_dev, lambda, alpha, i,
                                                 theta_c, crash_status) -
                       running_cost) /
                      (1.0 * i);
    }
    __syncthreads();

    // Increment states
    dynamics->step(state, state_next, state_der, control, output, theta_s, i, dt);
    __syncthreads();
    state_temp = state;
    state = state_next;
    state_next = state_temp;
  }
  // End loop outer loop on timesteps

  if (tdy == 0)
  {  // Only save the costs once per global idx (thread y is only for parallelization)
    costs_d[global_idx] = running_cost + costs->terminalCost(output, theta_c) / (num_timesteps - 1);
  }
}

// Newly Written
template <class DYN_T, class COST_T, class FB_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int NUM_ROLLOUTS, int BLOCKSIZE_Z>
__global__ void RMPPIRolloutKernel(DYN_T* dynamics, COST_T* costs, FB_T* fb_controller, float dt, int num_timesteps,
                                   int optimization_stride, float lambda, float alpha, float value_func_threshold,
                                   float* x_d, float* u_d, float* du_d, float* sigma_u_d, float* trajectory_costs_d)
{
  int thread_idx = threadIdx.x;
  int thread_idy = threadIdx.y;
  int thread_idz = threadIdx.z;
  int block_idx = blockIdx.x;
  int global_idx = BLOCKSIZE_X * block_idx + thread_idx;

  // Create shared memory for state and control
  __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float x_next_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float y_shared[BLOCKSIZE_X * DYN_T::OUTPUT_DIM * BLOCKSIZE_Z];
  __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float sigma_u[DYN_T::CONTROL_DIM];

  // Create a shared array for the nominal costs calculations
  __shared__ float running_state_cost_nom_shared[BLOCKSIZE_X];
  __shared__ float running_control_cost_nom_shared[BLOCKSIZE_X];
  __shared__ int crash_status_shared[BLOCKSIZE_X * BLOCKSIZE_Z];

  // Create a shared array for the dynamics model to use
  __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 +
                           DYN_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X * BLOCKSIZE_Z];
  __shared__ float theta_c[COST_T::SHARED_MEM_REQUEST_GRD + COST_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X * BLOCKSIZE_Z];

  // Create a shared array for the feedback controller to use
  __shared__ float theta_fb[FB_T::SHARED_MEM_SIZE];

  // Create local state, state dot and controls
  float* x;
  float* x_next;
  float* x_next_other;
  float* x_temp;
  float* y;
  float* x_other;
  float* xdot;
  float* u;
  float* du;
  int* crash_status;
  // The array to hold K(x,x*)
  float fb_control[DYN_T::CONTROL_DIM];

  int t = 0;
  int i = 0;
  // int j = 0;

  // Initialize running costs
  float running_state_cost_real = 0;
  float running_control_cost_real = 0;
  float* running_state_cost_nom;
  float running_tracking_cost_real = 0;
  float* running_control_cost_nom;

  // Load global array into shared memory
  if (global_idx < NUM_ROLLOUTS)
  {
    // Actual or nominal
    x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    x_next = &x_next_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    y = &y_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::OUTPUT_DIM];
    // The opposite state from above
    x_other = &x_shared[(blockDim.x * (1 - thread_idz) + thread_idx) * DYN_T::STATE_DIM];
    x_next_other = &x_next_shared[(blockDim.x * (1 - thread_idz) + thread_idx) * DYN_T::STATE_DIM];
    xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    // Base trajectory
    u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    // Noise added to trajectory
    du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    // Nominal State Cost
    running_state_cost_nom = &running_state_cost_nom_shared[thread_idx];
    running_control_cost_nom = &running_control_cost_nom_shared[thread_idx];
    crash_status = &crash_status_shared[thread_idz * blockDim.x + thread_idx];
    crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.

    // Load memory into appropriate arrays
    mppi_common::loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(
        NUM_ROLLOUTS, BLOCKSIZE_Y, global_idx, thread_idy, thread_idz, x_d, sigma_u_d, x, xdot, u, du, sigma_u);
    __syncthreads();
    *running_state_cost_nom = 0;
    *running_control_cost_nom = 0;
    float curr_state_cost = 0.0;
    dynamics->initializeDynamics(x, u, y, theta_s, 0.0, dt);
    costs->initializeCosts(y, u, theta_c, 0.0, dt);
    for (t = 0; t < num_timesteps; t++)
    {
      mppi_common::injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps, t, global_idx,
                                      thread_idy, optimization_stride, u_d, du_d, sigma_u, u, du);
      __syncthreads();

      // Now find feedback control
      for (i = 0; i < DYN_T::CONTROL_DIM; i++)
      {
        fb_control[i] = 0;
      }

      // we do not apply feedback on the nominal state z == 1
      if (thread_idz == 0)
      {
        fb_controller->k(x, x_other, t, theta_fb, fb_control);
      }

      for (i = thread_idy; i < DYN_T::CONTROL_DIM; i += BLOCKSIZE_Y)
      {
        u[i] += fb_control[i];
        // Make sure feedback is added to the modified control noise pointer
        // du_d[control_index + i] += fb_control[i];
      }

      __syncthreads();
      // Clamp the control in both the importance sampling sequence and the disturbed sequence.
      dynamics->enforceConstraints(x, u);
      __syncthreads();

      // Calculate All the costs
      if (t > 0)
      {
        curr_state_cost = costs->computeStateCost(y, t, theta_c, crash_status);
      }

      // Nominal system is where thread_idz == 1
      if (thread_idz == 1 && thread_idy == 0 && t > 0)
      {
        // This memory is shared in the y direction so limit which threads can write to it
        *running_state_cost_nom += curr_state_cost;
        *running_control_cost_nom += costs->computeLikelihoodRatioCost(u, du, sigma_u, lambda, alpha);
      }
      // Real system cost update when thread_idz == 0
      if (thread_idz == 0 && t > 0)
      {
        running_state_cost_real += curr_state_cost;
        running_control_cost_real += costs->computeLikelihoodRatioCost(u, du, sigma_u, lambda, alpha);

        running_tracking_cost_real +=
            (curr_state_cost + costs->computeFeedbackCost(fb_control, sigma_u, lambda, alpha));
      }

      __syncthreads();
      // Dynamics update
      dynamics->step(x, x_next, xdot, u, y, theta_s, t, dt);
      x_temp = x;
      x = x_next;
      x_next = x_temp;
      x_temp = x_other;
      x_other = x_next_other;
      x_next_other = x_temp;
    }

    // Compute average cost per timestep
    if (thread_idz == 1 && thread_idy == 0)
    {
      *running_state_cost_nom /= ((float)num_timesteps - 1);
      *running_control_cost_nom /= ((float)num_timesteps - 1);
    }

    if (thread_idz == 0)
    {
      running_state_cost_real /= ((float)num_timesteps - 1);
      running_tracking_cost_real /= ((float)num_timesteps - 1);
      running_control_cost_real /= ((float)num_timesteps - 1);
    }

    // calculate terminal costs
    if (thread_idz == 1 && thread_idy == 0)
    {  // Thread y required to prevent double addition
      *running_state_cost_nom += costs->terminalCost(y, theta_c) / (num_timesteps - 1);
    }

    if (thread_idz == 0)
    {
      running_state_cost_real += costs->terminalCost(y, theta_c) / (num_timesteps - 1);
      running_tracking_cost_real += costs->terminalCost(y, theta_c) / (num_timesteps - 1);
    }

    // Figure out final form of nominal cost
    __syncthreads();
    float running_cost_nom = 0;
    if (thread_idz == 0)
    {
      running_cost_nom = 0.5 * (*running_state_cost_nom) +
                         0.5 * fmaxf(fminf(running_tracking_cost_real, value_func_threshold), *running_state_cost_nom);

      running_cost_nom += *running_control_cost_nom;

      // Copy costs over to global memory
      // Actual System cost
      trajectory_costs_d[global_idx] = running_state_cost_real + running_control_cost_real;
      // Nominal System Cost - Again this is actaully only  known on real system threads
      trajectory_costs_d[global_idx + NUM_ROLLOUTS] = running_cost_nom;
    }
  }
}

/*******************************************************************************************************************
 * Launch Functions
 *******************************************************************************************************************/

template <class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int SAMPLES_PER_CONDITION>
void launchInitEvalKernel(DYN_T* dynamics, COST_T* costs, int num_candidates, int num_timesteps, float lambda,
                          float alpha, int ctrl_stride, float dt, int* strides_d, float* exploration_std_dev_d,
                          float* states_d, float* control_d, float* control_noise_d, float* costs_d,
                          cudaStream_t stream, bool synchronize)
{
  int GRIDSIZE_X = num_candidates * SAMPLES_PER_CONDITION / BLOCKSIZE_X;
  static_assert(SAMPLES_PER_CONDITION % BLOCKSIZE_X == 0,
                "SAMPLES_PER_CONDITION must be evenly divided by "
                "BLOCKSIZE_X");
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  dim3 dimGrid(GRIDSIZE_X, 1, 1);
  initEvalKernel<DYN_T, COST_T, BLOCKSIZE_X, BLOCKSIZE_Y, SAMPLES_PER_CONDITION>
      <<<dimGrid, dimBlock, 0, stream>>>(dynamics, costs, num_timesteps, lambda, alpha, ctrl_stride, dt, strides_d,
                                         exploration_std_dev_d, states_d, control_d, control_noise_d, costs_d);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, class COST_T, class FB_T, int NUM_ROLLOUTS, int BLOCKSIZE_X, int BLOCKSIZE_Y, int BLOCKSIZE_Z>
void launchRMPPIRolloutKernel(DYN_T* dynamics, COST_T* costs, FB_T* fb_controller, float dt, int num_timesteps,
                              int optimization_stride, float lambda, float alpha, float value_func_threshold,
                              float* x_d, float* u_d, float* du_d, float* sigma_u_d, float* trajectory_costs,
                              cudaStream_t stream, bool synchronize)
{
  const int gridsize_x = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;
  static_assert(NUM_ROLLOUTS % BLOCKSIZE_X == 0, "NUM_ROLLOUTS must be evenly divided by BLOCKSIZE_X");
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  dim3 dimGrid(gridsize_x, 1, 1);
  RMPPIRolloutKernel<DYN_T, COST_T, FB_T, BLOCKSIZE_X, BLOCKSIZE_Y, NUM_ROLLOUTS, BLOCKSIZE_Z>
      <<<dimGrid, dimBlock, 0, stream>>>(dynamics, costs, fb_controller, dt, num_timesteps, optimization_stride, lambda,
                                         alpha, value_func_threshold, x_d, u_d, du_d, sigma_u_d, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}
}  // namespace rmppi_kernels
