#include <mppi/core/mppi_common_new.cuh>

namespace mp1 = mppi::p1;

namespace mppi
{
namespace kernels
{
namespace rmppi
{
template <class DYN_T, class SAMPLING_T>
__global__ void initEvalDynKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling, float dt,
                                  const int num_timesteps, const int num_rollouts, int samples_per_condition,
                                  const int* __restrict__ strides_d, const float* __restrict__ states_d,
                                  float* __restrict__ y_d)
{
  // Get thread and block id
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int distribution_idx = threadIdx.z;
  const int candidate_idx = global_idx / samples_per_condition;
  const int candidate_sample_idx = global_idx % samples_per_condition;
  const int tdy = threadIdx.y;
  const int size_of_theta_s_bytes =
      math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  // const int thread_idx = threadIdx.x;
  // const int thread_idy = threadIdx.y;
  // const int thread_idz = threadIdx.z;
  // const int block_idx = blockIdx.x;
  // const int global_idx = blockDim.x * block_idx + thread_idx;
  const int shared_idx = blockDim.x * threadIdx.z + threadIdx.x;
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
  // int* strides_shared = (int*)&u_shared[math::nearest_multiple_4(sample_dim * DYN_T::CONTROL_DIM *
  // distribution_dim)]; float* theta_s_shared = (float*)&strides_shared[math::nearest_multiple_4(num_candidates)];
  // float* theta_s_shared = &u_shared[math::nearest_multiple_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_DYN
  barrier* barrier_shared = (barrier*)&theta_d_shared[size_of_theta_d_bytes / sizeof(float)];
#endif

  // Create local state, state dot and controls
  float* x = &(reinterpret_cast<float*>(x_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_next = &(reinterpret_cast<float*>(x_next_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_temp;
  float* xdot = &(reinterpret_cast<float*>(x_dot_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* u = &(reinterpret_cast<float*>(u_shared)[shared_idx * DYN_T::CONTROL_DIM]);
  float* y = &(reinterpret_cast<float*>(y_shared)[shared_idx * DYN_T::OUTPUT_DIM]);

  // Load global array to shared array
  // const int blocksize_y = blockDim.y;
  // loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(num_rollouts, blockDim.y, global_idx, tdy,
  // thread_idz,
  //                                                          init_x_d, x, xdot, u);

  mppi::p1::loadArrayParallel<DYN_T::STATE_DIM>(x, 0, states_d, candidate_idx * DYN_T::STATE_DIM);
  // mppi::p1::loadArrayParallel<mppi::p1::Parallel1Dir::THREAD_X>(strides_shared, 0, strides_d, 0, num_candidates);
  int stride = strides_d[candidate_idx];
  for (int i = tdy; i < DYN_T::CONTROL_DIM; i += blockDim.y)
  {
    u[i] = 0;
  }
  for (int i = tdy; i < DYN_T::OUTPUT_DIM; i += blockDim.y)
  {
    y[i] = 0;
  }
  __syncthreads();

#ifdef USE_CUDA_BARRIERS_DYN
  barrier* bar = &barrier_shared[(shared_idx)];
  if (tdy == 0)
  {
    init(bar, blockDim.y);
  }
#endif
  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0, dt);
  sampling->initializeDistributions(y, 0.0, dt, theta_d_shared);
  __syncthreads();
  for (int t = 0; t < num_timesteps; t++)
  {
    // __syncthreads();
    // Load noise trajectories scaled by the exploration factor
    int candidate_t = min(t + stride, num_timesteps - 1);
    sampling->readControlSample(candidate_sample_idx, candidate_t, distribution_idx, u, theta_d_shared, blockDim.y, tdy,
                                y);
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
    barrier::arrival_token control_read_token = bar->arrive();
    bar->wait(std::move(control_read_token));
#else
    __syncthreads();
#endif

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
    dynamics->enforceConstraints(x, u);
#ifdef USE_CUDA_BARRIERS_DYN
    barrier::arrival_token enforce_constraints_token = bar->arrive();
    bar->wait(std::move(enforce_constraints_token));
#else
    __syncthreads();
#endif

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
#ifdef USE_CUDA_BARRIERS_DYN
    barrier::arrival_token dynamics_token = bar->arrive();
    bar->wait(std::move(dynamics_token));
#else
    __syncthreads();
#endif
    x_temp = x;
    x = x_next;
    x_next = x_temp;
    // Copy state to global memory
    int sample_time_offset = (num_rollouts * threadIdx.z + global_idx) * num_timesteps + t;
    mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y_d, sample_time_offset * DYN_T::OUTPUT_DIM, y, 0);
  }
}

template <class COST_T, class SAMPLING_T, int BLOCKSIZE_X, bool COALESCE>
__global__ void initEvalCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                   const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                   const int samples_per_condition, const int* __restrict__ strides_d,
                                   const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int global_idx = blockIdx.x;
  const int distribution_idx = threadIdx.z;
  const int candidate_idx = global_idx / samples_per_condition;
  const int candidate_sample_idx = global_idx % samples_per_condition;
  // const int tdy = threadIdx.y;

  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  // const int global_idx = blockIdx.x;
  // const int distribution_idx = threadIdx.z;
  const int shared_idx = blockDim.x * threadIdx.z + threadIdx.x;
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
  int* crash_status_shared = (int*)&running_cost_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * blockDim.y)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  // int* strides_shared = &crash_status_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  // float* theta_c = (float*)&strides_shared[math::nearest_multiple_4(num_candidates)];
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
  int stride = strides_d[candidate_idx];

  // Load global array to shared array
  y = &y_shared[(shared_idx)*COST_T::OUTPUT_DIM];
  u = &u_shared[(shared_idx)*COST_T::CONTROL_DIM];
  crash_status = &crash_status_shared[shared_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  const int running_cost_index = thread_idx + blockDim.x * (thread_idy + blockDim.y * thread_idz);
  running_cost = &running_cost_shared[running_cost_index];
  running_cost[0] = 0;
#ifdef USE_CUDA_BARRIERS_COST
  barrier* bar = &barrier_shared[(blockDim.x * thread_idz + thread_idx)];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  /*<----Start of simulation loop-----> */
  // mppi::p1::loadArrayParallel<mppi::p1::Parallel1Dir::THREAD_X>(strides_shared, 0, strides_d, 0, num_candidates);
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
      int candidate_t = min(t + stride, num_timesteps - 1);
      // int candidate_t = min(t + strides_shared[candidate_idx], num_timesteps - 1);
      sampling->readControlSample(candidate_sample_idx, candidate_t, distribution_idx, u, theta_d, blockDim.y,
                                  thread_idy, y);
    }
#ifdef USE_CUDA_BARRIERS_COST
    barrier::arrival_token read_global_token = bar->arrive();
    bar->wait(std::move(read_global_token));
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
    barrier::arrival_token calc_cost_token = bar->arrive();
    bar->wait(std::move(calc_cost_token));
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

template <class DYN_T, class FB_T, class SAMPLING_T, int NOMINAL_STATE_IDX>
__global__ void rolloutRMPPIDynamicsKernel(DYN_T* __restrict__ dynamics, FB_T* __restrict__ fb_controller,
                                           SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                                           const int num_rollouts, const float* __restrict__ init_x_d,
                                           float* __restrict__ y_d)
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
  const int size_of_theta_s_bytes =
      math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim * math::int_multiple_const(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim *
          math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_fb_bytes =
      math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim * math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

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
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];
  float* theta_fb = &theta_d_shared[size_of_theta_d_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_DYN
  barrier* barrier_shared = (barrier*)&theta_fb[size_of_theta_fb_bytes / sizeof(float)];
#endif

  // Create local state, state dot and controls
  float* x = &(reinterpret_cast<float*>(x_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_next = &(reinterpret_cast<float*>(x_next_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* x_nom = &(reinterpret_cast<float*>(x_shared)[shared_nom_idx * DYN_T::STATE_DIM]);
  float* x_nom_next = &(reinterpret_cast<float*>(x_next_shared)[shared_nom_idx * DYN_T::STATE_DIM]);
  float* x_temp;
  float* xdot = &(reinterpret_cast<float*>(x_dot_shared)[shared_idx * DYN_T::STATE_DIM]);
  float* u = &(reinterpret_cast<float*>(u_shared)[shared_idx * DYN_T::CONTROL_DIM]);
  float* y = &(reinterpret_cast<float*>(y_shared)[shared_idx * DYN_T::OUTPUT_DIM]);
#ifdef USE_CUDA_BARRIERS_DYN
  barrier* bar = &barrier_shared[shared_idx];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif
  // The array to hold K(x,x*)
  float fb_control[DYN_T::CONTROL_DIM];
  int i;

  // Load global array to shared array
  ::mppi::kernels::loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(
      num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz, init_x_d, x, xdot, u);
  __syncthreads();

  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0, dt);
  sampling->initializeDistributions(y, 0.0, dt, theta_d_shared);
  fb_controller->initializeFeedback(x, u, theta_fb, 0.0, dt);
  __syncthreads();
  for (int t = 0; t < num_timesteps; t++)
  {
    // Load noise trajectories scaled by the exploration factor
    sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);
#ifdef USE_CUDA_BARRIERS_DYN
    barrier::arrival_token control_read_token = bar->arrive();
    bar->wait(std::move(control_read_token));
#else
    __syncthreads();
#endif

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
    }
#ifdef USE_CUDA_BARRIERS_DYN
    barrier::arrival_token feedback_token = bar->arrive();
    bar->wait(std::move(feedback_token));
#else
    __syncthreads();
#endif

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
    dynamics->enforceConstraints(x, u);
#ifdef USE_CUDA_BARRIERS_DYN
    barrier::arrival_token enforce_constraints_token = bar->arrive();
    bar->wait(std::move(enforce_constraints_token));
#else
    __syncthreads();
#endif
    // copy back feedback-filled controls to global memory
    sampling->writeControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
#ifdef USE_CUDA_BARRIERS_DYN
    barrier::arrival_token dynamics_token = bar->arrive();
    bar->wait(std::move(dynamics_token));
#else
    __syncthreads();
#endif
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

template <class COST_T, class DYN_T, class SAMPLING_T, class FB_T, int NOMINAL_STATE_IDX, bool COALESCE>
__global__ void rolloutRMPPICostKernel(COST_T* __restrict__ costs, DYN_T* __restrict__ dynamics,
                                       FB_T* __restrict__ fb_controller, SAMPLING_T* __restrict__ sampling, float dt,
                                       const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                       float value_func_threshold, const float* __restrict__ init_x_d,
                                       const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;
  const int distribution_idx = threadIdx.z;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int num_shared = blockDim.x * blockDim.z;
  const int size_of_theta_c_bytes =
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      num_shared * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_d_bytes =
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  const int size_of_theta_fb_bytes =
      math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      num_shared * math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];
  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[math::nearest_multiple_4(num_shared * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[math::nearest_multiple_4(num_shared * COST_T::CONTROL_DIM)];
  int* crash_status_shared = (int*)&running_cost_shared[math::nearest_multiple_4(num_shared * 2 * blockDim.y)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_multiple_4(num_shared)];
  float* theta_d = &theta_c[size_of_theta_c_bytes / sizeof(float)];
  float* theta_fb = &theta_d[size_of_theta_d_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_COST
  barrier* barrier_shared = (barrier*)&theta_fb[size_of_theta_fb_bytes / sizeof(float)];
#endif

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
  const int cost_index = blockDim.x * (thread_idz * blockDim.y + thread_idy) + thread_idx;
  float* running_cost = &running_cost_shared[cost_index];
  float* running_cost_extra = &running_cost_shared[cost_index + num_shared * blockDim.y];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  running_cost[0] = 0;
  running_cost_extra[0] = 0;
#ifdef USE_CUDA_BARRIERS_COST
  barrier* bar = &barrier_shared[(blockDim.x * thread_idz + thread_idx)];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  float curr_cost = 0.0f;

  /*<----Start of simulation loop-----> */
  const int max_time_iters = ceilf((float)num_timesteps / blockDim.x);
  costs->initializeCosts(y, u, theta_c, 0.0, dt);
  sampling->initializeDistributions(y, 0.0, dt, theta_d);
  fb_controller->initializeFeedback(y, u, theta_fb, 0.0, dt);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x + 1;  // start at t = 1
    if (t <= num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      if (COALESCE)
      {  // Fill entire shared mem sequentially using sequential threads_idx
        mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_X>(
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
    float x[DYN_T::STATE_DIM];
    float x_nom[DYN_T::STATE_DIM];
    dynamics->outputToState(y, x);
    dynamics->outputToState(y_nom, x_nom);
#ifdef USE_CUDA_BARRIERS_COST
    barrier::arrival_token read_global_token = bar->arrive();
    bar->wait(std::move(read_global_token));
#else
    __syncthreads();
#endif
    fb_controller->k(x, x_nom, t, theta_fb, fb_control);
#ifdef USE_CUDA_BARRIERS_COST
    barrier::arrival_token feedback_token = bar->arrive();
    bar->wait(std::move(feedback_token));
#else
    __syncthreads();
#endif

    // Compute cost
    if (t < num_timesteps)
    {
      curr_cost = costs->computeRunningCost(y, u, t, theta_c, crash_status);
    }

    if (thread_idz == NOMINAL_STATE_IDX && t < num_timesteps)
    {
      running_cost[0] += curr_cost;
      running_cost_extra[0] +=
          sampling->computeLikelihoodRatioCost(u, theta_d, global_idx, t, distribution_idx, lambda, alpha);
    }

    if (thread_idz != NOMINAL_STATE_IDX && t < num_timesteps)
    {
      running_cost[0] +=
          curr_cost + sampling->computeLikelihoodRatioCost(u, theta_d, global_idx, t, distribution_idx, lambda, alpha);
      running_cost_extra[0] +=
          curr_cost + sampling->computeFeedbackCost(fb_control, theta_d, t, distribution_idx, lambda, alpha);
    }

    // We need running_state_cost_nom and running_control_cost_nom for the nominal system
    // We need running_cost_real and running_cost_feedback for the real system
    // Nominal system needs to know running_state_cost_nom, running_control_cost_nom, and running_cost_feedback
    // Real system needs to know running_cost_real

#ifdef USE_CUDA_BARRIERS_COST
    barrier::arrival_token calc_cost_token = bar->arrive();
    bar->wait(std::move(calc_cost_token));
#else
    __syncthreads();
#endif
  }

  // Add all costs together
  running_cost = &running_cost_shared[blockDim.x * blockDim.y * thread_idz];
  running_cost_extra = &running_cost_shared[blockDim.x * blockDim.y * (blockDim.z + thread_idz)];

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
      running_cost_extra[j] += running_cost_extra[j + size];
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
  if (xy_index < 32)
  {  // unroll the last warp
    switch (size * 2)
    {
      case 64:
        warpReduceAdd<64>(running_cost, xy_index);
        warpReduceAdd<64>(running_cost_extra, xy_index);
        break;
      case 32:
        warpReduceAdd<32>(running_cost, xy_index);
        warpReduceAdd<32>(running_cost_extra, xy_index);
        break;
      case 16:
        warpReduceAdd<16>(running_cost, xy_index);
        warpReduceAdd<16>(running_cost_extra, xy_index);
        break;
      case 8:
        warpReduceAdd<8>(running_cost, xy_index);
        warpReduceAdd<8>(running_cost_extra, xy_index);
        break;
      case 4:
        warpReduceAdd<4>(running_cost, xy_index);
        warpReduceAdd<4>(running_cost_extra, xy_index);
        break;
      case 2:
        warpReduceAdd<2>(running_cost, xy_index);
        warpReduceAdd<2>(running_cost_extra, xy_index);
        break;
      case 1:
        warpReduceAdd<1>(running_cost, xy_index);
        warpReduceAdd<1>(running_cost_extra, xy_index);
        break;
    }
  }

  __syncthreads();
  // running_cost = &running_cost_shared[blockDim.x * thread_idz];
  // running_cost_extra = &running_cost_shared[blockDim.x * (blockDim.z + thread_idz)];
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % blockDim.x;
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
  if (thread_idz != NOMINAL_STATE_IDX && thread_idx == 0 && thread_idy == 0)
  {
    float* running_nom_cost = &running_cost_shared[blockDim.x * blockDim.y * NOMINAL_STATE_IDX];
    const float* running_nom_cost_likelihood_ratio_cost =
        &running_cost_shared[blockDim.x * blockDim.y * (blockDim.z + NOMINAL_STATE_IDX)];
    // const float value_func_threshold = 109;
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
  // theta_c, trajectory_costs_d);
}

template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchFastInitEvalKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs,
                              SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                              const int num_rollouts, float lambda, float alpha, int samples_per_condition,
                              int* __restrict__ strides_d, float* __restrict__ init_x_d, float* __restrict__ y_d,
                              float* __restrict__ trajectory_costs, dim3 dimDynBlock, dim3 dimCostBlock,
                              cudaStream_t stream, bool synchronize)
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

  initEvalDynKernel<DYN_T, SAMPLING_T><<<dimGrid, dimDynBlock, dynamics_shared_size, stream>>>(
      dynamics, sampling, dt, num_timesteps, num_rollouts, samples_per_condition, strides_d, init_x_d, y_d);
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
      cost_num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      math::nearest_multiple_4(num_rollouts / samples_per_condition) * sizeof(float);
#ifdef USE_CUDA_BARRIERS_COST
  cost_shared_size += math::int_multiple_const(cost_num_shared * sizeof(barrier), 16);
#endif

  initEvalCostKernel<COST_T, SAMPLING_T, COST_BLOCK_X><<<dimCostGrid, dimCostBlock, cost_shared_size, stream>>>(
      costs, sampling, dt, num_timesteps, num_rollouts, lambda, alpha, samples_per_condition, strides_d, y_d,
      trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, class COST_T, class SAMPLING_T, class FB_T, int NOMINAL_STATE_IDX>
void launchFastRMPPIRolloutKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs,
                                  SAMPLING_T* __restrict__ sampling, FB_T* __restrict__ fb_controller, float dt,
                                  const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                  float value_func_threshold, float* __restrict__ init_x_d, float* __restrict__ y_d,
                                  float* __restrict__ trajectory_costs, dim3 dimDynBlock, dim3 dimCostBlock,
                                  cudaStream_t stream, bool synchronize)
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
  if (dimDynBlock.z < 2 || dimDynBlock.z != dimCostBlock.z)
  {
    std::cerr << __FILE__ << " (" << __LINE__ << "): number of dynamics systems (" << dimDynBlock.z
              << ") and cost systems (" << dimCostBlock.z << ") must be equal to each other and greater than 2"
              << std::endl;
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

  dynamics_shared_size +=
      math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
#ifdef USE_CUDA_BARRIERS_DYN
  dynamics_shared_size += math::int_multiple_const(dynamics_num_shared * sizeof(barrier), 16);
#endif

  rolloutRMPPIDynamicsKernel<DYN_T, FB_T, SAMPLING_T, NOMINAL_STATE_IDX>
      <<<dimGrid, dimDynBlock, dynamics_shared_size, stream>>>(dynamics, fb_controller, sampling, dt, num_timesteps,
                                                               num_rollouts, init_x_d, y_d);
  // Run Costs
  dim3 dimCostGrid(num_rollouts, 1, 1);
  const int cost_num_shared = dimCostBlock.x * dimCostBlock.z;
  unsigned cost_shared_size =
      sizeof(float) * (math::nearest_multiple_4(cost_num_shared * COST_T::OUTPUT_DIM) +
                       math::nearest_multiple_4(cost_num_shared * COST_T::CONTROL_DIM) +
                       math::nearest_multiple_4(cost_num_shared * dimCostBlock.y * 2)) +
      sizeof(int) * math::nearest_multiple_4(cost_num_shared) +
      math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      cost_num_shared * math::int_multiple_const(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      cost_num_shared * math::int_multiple_const(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  cost_shared_size += math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
                      cost_num_shared * math::int_multiple_const(FB_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
#ifdef USE_CUDA_BARRIERS_COST
  cost_shared_size += math::int_multiple_const(cost_num_shared * sizeof(barrier), 16);
#endif
  rolloutRMPPICostKernel<COST_T, DYN_T, SAMPLING_T, FB_T, NOMINAL_STATE_IDX>
      <<<dimCostGrid, dimCostBlock, cost_shared_size, stream>>>(costs, dynamics, fb_controller, sampling, dt,
                                                                num_timesteps, num_rollouts, lambda, alpha,
                                                                value_func_threshold, init_x_d, y_d, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize || true)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
  HANDLE_ERROR(cudaGetLastError());
}
}  // namespace rmppi
}  // namespace kernels
}  // namespace mppi
