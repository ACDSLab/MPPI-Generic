#include <mppi/utils/math_utils.h>
#include <mppi/core/mppi_common.cuh>

namespace mp1 = mppi::p1;

namespace mppi
{
namespace kernels
{
template <class COST_T, class SAMPLING_T, int BLOCKSIZE_X, bool COALESCE>
__global__ void rolloutCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                  const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                  const float* __restrict__ init_x_d, const float* __restrict__ y_d,
                                  float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;
  const int distribution_idx = threadIdx.z;

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];
  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::CONTROL_DIM)];
  int* crash_status_shared = (int*)&running_cost_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  const int size_of_theta_c_bytes =
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  float* theta_d = &theta_c[size_of_theta_c_bytes / sizeof(float)];

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
  running_cost = &running_cost_shared[thread_idz * blockDim.x + thread_idx];
  running_cost[0] = 0;

  /*<----Start of simulation loop-----> */
  const int max_time_iters = ceilf((float)num_timesteps / blockDim.x);
  costs->initializeCosts(y, u, theta_c, 0.0, dt);
  sampling->initializeDistributions(y, 0.0, dt, theta_d);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x + 1;  // start at t = 1
    if (t <= num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      if (COALESCE)
      {  // Fill entire shared mem sequentially using sequential threads_idx
        mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_X>(
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
      sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d, blockDim.y, thread_idy, y);
    }
    __syncthreads();

    // Compute cost
    if (thread_idy == 0 && t < num_timesteps)
    {
      running_cost[0] += costs->computeRunningCost(y, u, t, theta_c, crash_status) +
                         sampling->computeLikelihoodRatioCost(u, theta_d, t, distribution_idx, lambda, alpha);
    }
    __syncthreads();
  }

  // Add all costs together
  running_cost = &running_cost_shared[blockDim.x * thread_idz];
#if true
  int prev_size = blockDim.x;
  for (int size = prev_size / 2; size > 32; size /= 2)
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
  if (thread_idx < 32 && thread_idy == 0)
  {  // unroll the last warp
    switch (blockDim.x)
    {
      case 64:
        warpReduceAdd<64>(running_cost, thread_idx);
        break;
      case 32:
        warpReduceAdd<32>(running_cost, thread_idx);
        break;
      case 16:
        warpReduceAdd<16>(running_cost, thread_idx);
        break;
      case 8:
        warpReduceAdd<8>(running_cost, thread_idx);
        break;
      case 4:
        warpReduceAdd<4>(running_cost, thread_idx);
        break;
      case 2:
        warpReduceAdd<2>(running_cost, thread_idx);
        break;
      case 1:
        warpReduceAdd<1>(running_cost, thread_idx);
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
  const int last_y_index = (num_timesteps - 1) % BLOCKSIZE_X;
  y = &y_shared[(blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM];
  // Compute terminal cost and the final cost for each thread
  mppi_common::computeAndSaveCost(num_rollouts, num_timesteps, global_idx, costs, y,
                                  running_cost[0] / (num_timesteps - 1), theta_c, trajectory_costs_d);
}

template <class DYN_T, class SAMPLING_T>
__global__ void rolloutDynamicsKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling, float dt,
                                      const int num_timesteps, const int optimization_stride, const int num_rollouts,
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

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* x_shared = entire_buffer;
  float* x_next_shared = &x_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[math::nearest_multiple_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[math::nearest_multiple_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  const int size_of_theta_s_bytes =
      math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim * math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];

  // Create local state, state dot and controls
  float* x = &(reinterpret_cast<float*>(x_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_next = &(reinterpret_cast<float*>(x_next_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_temp;
  float* xdot = &(reinterpret_cast<float*>(x_dot_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* u = &(reinterpret_cast<float*>(u_shared)[shared_idx * DYN_T::CONTROL_DIM]);
  float* y = &(reinterpret_cast<float*>(y_shared)[shared_idx * DYN_T::OUTPUT_DIM]);

  // Load global array to shared array
  // const int blocksize_y = blockDim.y;
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz,
                                                           init_x_d, x, xdot, u);
  __syncthreads();

  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0, dt);
  sampling->initializeDistributions(y, 0.0, dt, theta_d_shared);
  __syncthreads();
  for (int t = 0; t < num_timesteps; t++)
  {
    __syncthreads();
    dynamics->enforceConstraints(x, sampling->getControlSample(global_idx, t, distribution_idx, theta_d_shared, y));
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
    __syncthreads();

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
    // dynamics->enforceConstraints(x, u);
    __syncthreads();

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
    __syncthreads();
    x_temp = x;
    x = x_next;
    x_next = x_temp;
    // Copy state to global memory
    int sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t;
    mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y_d, sample_time_offset * DYN_T::OUTPUT_DIM, y, 0);
  }
}

template <class DYN_T, class FB_T, class SAMPLING_T, int NOMINAL_STATE_IDX = 0>
__global__ void rolloutRMPPIDynamicsKernel(DYN_T* __restrict__ dynamics, FB_T* __restrict__ fb_controller,
                                           SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                                           const int optimization_stride, const int num_rollouts,
                                           const float* __restrict__ init_x_d, float* __restrict__ y_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int block_idx = blockIdx.x;
  const int global_idx = blockDim.x * block_idx + thread_idx;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int shared_nom_idx = blockDim.x * NOMINAL_STATE_IDX + thread_idx;
  const int distribution_idx = threadIdx.z;
  const int distribution_dim = blockDim.z;
  const int sample_dim = blockDim.x;

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* x_shared = entire_buffer;
  float* x_next_shared = &x_shared[math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[math::nearest_quotient_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[math::nearest_quotient_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  const int size_of_theta_s_bytes =
      math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim * math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim *
          math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];
  float* theta_fb = &theta_d_shared[size_of_theta_d_bytes / sizeof(float)];

  // Create local state, state dot and controls
  float* x = &(reinterpret_cast<float*>(x_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_next = &(reinterpret_cast<float*>(x_next_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_nom = &(reinterpret_cast<float*>(x_shared)[shared_nom_idx * DYN_T::STATE_DIM]);
  float* x_nom_next = &(reinterpret_cast<float*>(x_next_shared)[shared_nom_idx * DYN_T::STATE_DIM]);
  float* x_temp;
  float* xdot = &(reinterpret_cast<float*>(x_dot_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* u = &(reinterpret_cast<float*>(u_shared)[shared_idx * DYN_T::CONTROL_DIM]);
  float* y = &(reinterpret_cast<float*>(y_shared)[shared_idx * DYN_T::OUTPUT_DIM]);
  // The array to hold K(x,x*)
  float fb_control[DYN_T::CONTROL_DIM];
  int i;

  // Load global array to shared array
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz,
                                                           init_x_d, x, xdot, u);
  __syncthreads();

  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0, dt);
  sampling->initializeDistributions(y, 0.0, dt, theta_d_shared);
  fb_controller->initiliazeFeedback(x, u, theta_fb, 0.0, dt);
  for (int t = 0; t < num_timesteps; t++)
  {
    // Load noise trajectories scaled by the exploration factor
    sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);
    __syncthreads();

    // Now find feedback control
    for (i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      fb_control[i] = 0;
    }

    // we do not apply feedback on the nominal state
    if (thread_idz != NOMINAL_STATE_IDX)
    {
      fb_controller->k(x, x_nom, t, theta_fb, fb_control);
    }

    for (i = thread_idy; i < DYN_T::CONTROL_DIM; i += blockDim.y)
    {
      u[i] += fb_control[i];
      // Make sure feedback is added to the modified control noise pointer
      // du_d[control_index + i] += fb_control[i];
    }
    __syncthreads();

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
    dynamics->enforceConstraints(x, u);
    __syncthreads();

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
    // copy back feedback-filled controls to global memory
    if (thread_idz != NOMINAL_STATE_IDX)
    {
      mp1::loadArrayParallel<DYN_T::CONTROL_DIM>(sampling->getControlSample(global_idx, t, distribution_idx, y), 0, u,
                                                 0);
    }
    __syncthreads();
    x_temp = x;
    x = x_next;
    x_next = x_temp;
    x_temp = x_nom;
    x_nom = x_nom_next;
    x_nom_next = x_temp;
    // Copy state to global memory
    int sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t;
    mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y_d, sample_time_offset * DYN_T::OUTPUT_DIM, y, 0);
  }
}

template <class COST_T, class SAMPLING_T, class FB_T, int BLOCKSIZE_X, bool COALESCE, int NOMINAL_STATE_IDX = 0>
__global__ void rolloutRMPPICostKernel(COST_T* __restrict__ costs, FB_T* __restrict__ fb_controller,
                                       SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                                       const int num_rollouts, float lambda, float alpha,
                                       const float* __restrict__ init_x_d, const float* __restrict__ y_d,
                                       float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;
  const int distribution_idx = threadIdx.z;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int num_shared = blockDim.x * blockDim.z;

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];
  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[math::nearest_quotient_4(num_shared * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[math::nearest_quotient_4(num_shared * COST_T::CONTROL_DIM)];
  int* crash_status_shared = (int*)&running_cost_shared[math::nearest_quotient_4(num_shared * 2)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_quotient_4(num_shared)];
  const int size_of_theta_c_bytes =
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      num_shared * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  float* theta_d = &theta_c[size_of_theta_c_bytes / sizeof(float)];

  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  float* theta_fb = &theta_d[size_of_theta_d_bytes / sizeof(float)];

  // Initialize running cost and total cost
  int sample_time_offset = 0;
  int j = 0;

  // The array to hold K(x,x*)
  float fb_control[COST_T::CONTROL_DIM];

  // Load global array to shared array
  float* y = &y_shared[shared_idx * COST_T::OUTPUT_DIM];
  float* y_nom = &y_shared[(blockDim.x * NOMINAL_STATE_IDX + thread_idx) * COST_T::OUTPUT_DIM];
  float* u = &u_shared[shared_idx * COST_T::CONTROL_DIM];
  int* crash_status = &crash_status_shared[shared_idx];
  float* running_cost = &running_cost_shared[shared_idx];
  float* running_cost_extra = &running_cost_shared[shared_idx + num_shared];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  running_cost[0] = 0;
  running_cost_extra[0] = 0;

  float curr_cost = 0.0f;

  /*<----Start of simulation loop-----> */
  const int max_time_iters = ceilf((float)num_timesteps / blockDim.x);
  costs->initializeCosts(y, u, theta_c, 0.0, dt);
  sampling->initializeDistributions(y, 0.0, dt, theta_d);
  fb_controller->initiliazeFeedback(y, u, theta_fb, 0.0, dt);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x + 1;  // start at t = 1
    if (t <= num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      if (COALESCE)
      {  // Fill entire shared mem sequentially using sequential threads_idx
        mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_X>(
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
      sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d, blockDim.y, thread_idy, y);
    }
    __syncthreads();

    // Compute cost
    if (thread_idy == 0 && t < num_timesteps)
    {
      curr_cost = costs->computeRunningCost(y, u, t, theta_c, crash_status);
    }

    if (thread_idz == NOMINAL_STATE_IDX && thread_idy == 0 && t < num_timesteps)
    {
      running_cost[0] += curr_cost;
      running_cost_extra[0] += sampling->computeLikelihoodRatioCost(u, theta_d, t, distribution_idx, lambda, alpha);
    }
    if (thread_idz != NOMINAL_STATE_IDX && thread_idy == 0 && t < num_timesteps)
    {
      // we do not apply feedback on the nominal state
      float x[COST_T::STATE_DIM];
      float x_nom[COST_T::STATE_DIM];
      // dynamics->outputToState(y, x);
      // dynamics->outputToState(y_nom, x_nom);
      __syncthreads();
      fb_controller->k(x, x_nom, t, theta_fb, fb_control);

      running_cost[0] +=
          curr_cost + sampling->computeLikelihoodRatioCost(u, theta_d, t, distribution_idx, lambda, alpha);
      running_cost_extra[0] = curr_cost + sampling->computeFeedbackCost(fb_control, lambda, alpha);
    }

    // We need running_state_cost_nom and running_control_cost_nom for the nominal system
    // We need running_cost_real and running_cost_feedback for the real system
    // Nominal system needs to know running_state_cost_nom, running_control_cost_nom, and running_cost_feedback
    // Real system needs to know running_cost_real

    __syncthreads();
  }

  // Add all costs together
  int prev_size = BLOCKSIZE_X;
  running_cost = &running_cost_shared[blockDim.x * thread_idz];
  running_cost_extra = &running_cost_shared[blockDim.x * (blockDim.z + thread_idz)];
#if true
  for (int size = prev_size / 2; size > 32; size /= 2)
  {
    if (thread_idy == 0)
    {
      for (j = thread_idx; j < size; j += blockDim.x)
      {
        running_cost[j] += running_cost[j + size];
        running_cost_extra[j] += running_cost_extra[j + size];
      }
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1 && thread_idy == 0)
    {
      running_cost[size - 1] += running_cost[prev_size - 1];
      running_cost_extra[size - 1] += running_cost_extra[prev_size - 1];
    }
    __syncthreads();
    prev_size = size;
  }
  if (thread_idx < 32 && thread_idy == 0)
  {  // unroll the last warp
    switch (blockDim.x)
    {
      case 64:
        warpReduceAdd<64>(running_cost, thread_idx);
        warpReduceAdd<64>(running_cost_extra, thread_idx);
        break;
      case 32:
        warpReduceAdd<32>(running_cost, thread_idx);
        warpReduceAdd<32>(running_cost_extra, thread_idx);
        break;
      case 16:
        warpReduceAdd<16>(running_cost, thread_idx);
        warpReduceAdd<16>(running_cost_extra, thread_idx);
        break;
      case 8:
        warpReduceAdd<8>(running_cost, thread_idx);
        warpReduceAdd<8>(running_cost_extra, thread_idx);
        break;
      case 4:
        warpReduceAdd<4>(running_cost, thread_idx);
        warpReduceAdd<4>(running_cost_extra, thread_idx);
        break;
      case 2:
        warpReduceAdd<2>(running_cost, thread_idx);
        warpReduceAdd<2>(running_cost_extra, thread_idx);
        break;
      case 1:
        warpReduceAdd<1>(running_cost, thread_idx);
        warpReduceAdd<1>(running_cost_extra, thread_idx);
        break;
    }
  }
#else
#pragma unroll
  for (int size = prev_size / 2; size > 0; size /= 2)
  {
    if (thread_idy == 0)
    {
      for (j = thread_idx; j < size; j += blockDim.x)
      {
        running_cost[j] += running_cost[j + size];
        running_cost_extra[j] += running_cost_extra[j + size];
      }
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1 && thread_idy == 0)
    {
      running_cost[size - 1] += running_cost[prev_size - 1];
      running_cost_extra[size - 1] += running_cost_extra[prev_size - 1];
    }
    __syncthreads();
    prev_size = size;
  }
#endif
  __syncthreads();
  // running_cost = &running_cost_shared[blockDim.x * thread_idz];
  // running_cost_extra = &running_cost_shared[blockDim.x * (blockDim.z + thread_idz)];
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % BLOCKSIZE_X;
  y = &y_shared[(blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM];
  if (thread_idx == 0 && thread_idy == 0)
  {
    *running_cost += costs->terminalCost(y, theta_c);
    if (thread_idz != NOMINAL_STATE_IDX)
    {
      *running_cost_extra += costs->terminalCost(y, theta_c);
    }
    *running_cost /= ((float)num_timesteps - 1);
    *running_cost_extra /= ((float)num_timesteps - 1);
  }
  __syncthreads();
  if (thread_idz == (1 - NOMINAL_STATE_IDX) && thread_idx == 0 && thread_idy == 0)
  {
    float* running_nom_cost = &running_cost_shared[blockDim.x * NOMINAL_STATE_IDX];
    const float* running_nom_cost_likelihood_ratio_cost =
        &running_cost_shared[blockDim.x * (blockDim.z + NOMINAL_STATE_IDX)];
    const float value_func_threshold = 109;
    *running_nom_cost =
        0.5 * *running_nom_cost + 0.5 * fmaxf(fminf(*running_cost_extra, value_func_threshold), *running_nom_cost);
    *running_nom_cost += *running_nom_cost_likelihood_ratio_cost;
  }
  __syncthreads();
  if (thread_idy == 0 && thread_idx == 0)
  {
    trajectory_costs_d[global_idx + thread_idz * num_rollouts] = running_cost[0];
  }
  // Compute terminal cost and the final cost for each thread
  // computeAndSaveCost(num_rollouts, num_timesteps, global_idx, costs, y, running_cost_extra[0] / (num_timesteps - 1),
  // theta_c,
  //                    trajectory_costs_d);
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
        xdot4_t[i] = make_float4(0, 0, 0, 0);
      }
    }
    else if (STATE_DIM % 2 == 0)
    {
      float2* xdot2_t = reinterpret_cast<float2*>(xdot_thread);
      for (i = thread_idy; i < STATE_DIM / 2; i += blocksize_y)
      {
        xdot2_t[i] = make_float2(0, 0);
      }
    }
    else
    {
      for (i = thread_idy; i < STATE_DIM; i += blocksize_y)
      {
        xdot_thread[i] = 0;
      }
    }

    if (CONTROL_DIM % 4 == 0)
    {
      float4* u4_t = reinterpret_cast<float4*>(u_thread);
      for (i = thread_idy; i < CONTROL_DIM / 4; i += blocksize_y)
      {
        u4_t[i] = make_float4(0, 0, 0, 0);
      }
    }
    else if (CONTROL_DIM % 2 == 0)
    {
      float2* u2_t = reinterpret_cast<float2*>(u_thread);
      for (i = thread_idy; i < CONTROL_DIM / 2; i += blocksize_y)
      {
        u2_t[i] = make_float2(0, 0);
      }
    }
    else
    {
      for (i = thread_idy; i < CONTROL_DIM; i += blocksize_y)
      {
        u_thread[i] = 0;
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
__device__ void warpReduceAdd(volatile float* sdata, const int tid)
{
  if (BLOCKSIZE >= 64)
  {
    sdata[tid] += sdata[tid + 32];
  }
  if (BLOCKSIZE >= 32)
  {
    sdata[tid] += sdata[tid + 16];
  }
  if (BLOCKSIZE >= 16)
  {
    sdata[tid] += sdata[tid + 8];
  }
  if (BLOCKSIZE >= 8)
  {
    sdata[tid] += sdata[tid + 4];
  }
  if (BLOCKSIZE >= 4)
  {
    sdata[tid] += sdata[tid + 2];
  }
  if (BLOCKSIZE >= 2)
  {
    sdata[tid] += sdata[tid + 1];
  }
}

template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchFastRolloutKernel(DYN_T* dynamics, COST_T* costs, SAMPLING_T* sampling, float dt, const int num_timesteps,
                             const int num_rollouts, const int optimization_stride, float lambda, float alpha,
                             float* __restrict__ init_x_d, float* __restrict__ y_d,
                             float* __restrict__ trajectory_costs, dim3 dimDynBlock, dim3 dimCostBlock,
                             cudaStream_t stream, bool synchronize)
{
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
      dynamics, sampling, dt, num_timesteps, optimization_stride, num_rollouts, init_x_d, y_d);
  // Run Costs
  dim3 dimCostGrid(num_rollouts, 1, 1);
  const int cost_num_shared = dimCostBlock.x * dimCostBlock.z;
  const int COST_BLOCK_X = 64;
  unsigned cost_shared_size =
      sizeof(float) * (math::nearest_multiple_4(cost_num_shared * COST_T::OUTPUT_DIM) +
                       math::nearest_multiple_4(cost_num_shared * COST_T::CONTROL_DIM) +
                       math::nearest_multiple_4(cost_num_shared)) +
      sizeof(int) * math::nearest_multiple_4(cost_num_shared) +
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  rolloutCostKernel<COST_T, SAMPLING_T, COST_BLOCK_X><<<dimCostGrid, dimCostBlock, cost_shared_size, stream>>>(
      costs, sampling, dt, num_timesteps, num_rollouts, lambda, alpha, init_x_d, y_d, trajectory_costs);
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
  unsigned shared_mem_size = math::nearest_quotient_4(CONTROL_DIM * dimBlock.x);
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
