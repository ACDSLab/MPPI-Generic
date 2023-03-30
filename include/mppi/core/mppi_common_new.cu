#include <mppi/utils/math_utils.h>

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

__global__ void weightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                        float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                        const int num_rollouts, const int sum_stride, const int control_dim)
{
  int thread_idx = threadIdx.x;  // Rollout index
  int block_idx = blockIdx.x;    // Timestep

  // Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
  extern __shared__ float u_intermediate[];

  float u[control_dim];
  setInitialControlToZero(control_dim, thread_idx, u, u_intermediate);

  __syncthreads();

  // Sum the weighted control variations at a desired stride
  strideControlWeightReduction(num_rollouts, num_timesteps, sum_stride, thread_idx, block_idx, control_dim, exp_costs_d,
                               normalizer, du_d, u, u_intermediate);

  __syncthreads();

  // Sum all weighted control variations
  rolloutWeightReductionAndSaveControl(thread_idx, block_idx, num_rollouts, num_timesteps, control_dim, sum_stride, u,
                                       u_intermediate, new_u_d);

  __syncthreads();
}

void launchWeightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                   float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                   const int num_rollouts, const int sum_stride, const int control_dim,
                                   cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock(mppi::math::int_ceil(num_rollouts, sum_stride), 1, 1);
  dim3 dimGrid(num_timesteps, 1, 1);
  unsigned shared_mem_size = mppi::math::nearest_quotient_4(control_dim * dimBlock.x);
  weightedReductionKernel<<<dimGrid, dimBlock, shared_mem_size, stream>>>(
      exp_costs_d, du_d, new_u_d, normalizer, num_timesteps, num_rollouts, sum_stride, control_dim);
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
  float* x_next_shared = &x_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  const int size_of_theta_s_bytes =
      mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      sample_dim * distribution_dim * mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];

  // Create local state, state dot and controls
  float* x;
  float* x_next;
  float* x_temp;
  float* xdot;
  float* u;
  float* y;

  // Load global array to shared array
  if (global_idx < num_rollouts)
  {
    x = &(reinterpret_cast<float*>(x_shared)[shared_idx * DYN_T::STATE_DIM]);
    x_next = &(reinterpret_cast<float*>(x_next_shared)[shared_idx * DYN_T::STATE_DIM]);
    y = &(reinterpret_cast<float*>(y_shared)[shared_idx * DYN_T::OUTPUT_DIM]);
    xdot = &(reinterpret_cast<float*>(xdot_shared)[shared_idx * DYN_T::STATE_DIM]);
    u = &(reinterpret_cast<float*>(u_shared)[shared_idx * DYN_T::CONTROL_DIM]);
  }

  loadGlobalToShared(num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz, init_x_d, x, xdot, u);
  __syncthreads();

  if (global_idx < num_rollouts)
  {
    /*<----Start of simulation loop-----> */
    dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0, dt);
    sampling->initializeDistributions(y, 0.0, dt, theta_d_shared);
    for (int t = 0; t < num_timesteps; t++)
    {
      // Load noise trajectories scaled by the exploration factor
      // injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, num_rollouts, num_timesteps, t, global_idx, thread_idy,
      //                    optimization_stride, u_d, du_d, reinterpret_cast<float*>(sigma_u), u, du);
      sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);
      // du_d is now v
      __syncthreads();

      // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
      // usually just control clamping
      // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU
      dynamics->enforceConstraints(x, sampling->getControlSample(global_idx, t, distribution_idx, y));
      dynamics->enforceConstraints(x, u);
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
}

template <class COST_T, class SAMPLING_T, int BLOCKSIZE_X, bool COALESCE = false>
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
  float* u_shared = &y_shared[mppi::math::nearest_quotient_4(blockDim.x * blockDim.z * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[mppi::math::nearest_quotient_4(blockDim.x * blockDim.z * COST_T::CONTROL_DIM)];
  int* crash_status_shared = (int*)&running_cost_shared[mppi::math::nearest_quotient_4(blockDim.x * blockDim.z)];
  float* theta_c = (float*)&crash_status_shared[mppi::math::nearest_quotient_4(blockDim.x * blockDim.z)];
  const int size_of_theta_c_bytes =
      mppi::math::int_ceil(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      blockDim.x * blockDim.z * mppi::math::int_ceil(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));
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
  sampling->initializeDistributions(s, 0.0, dt, theta_d);
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
      running_cost[0] += costs->computeRunningCost(y, u, du, sigma_u, lambda, alpha, t, theta_c, crash_status) +
                         sampling->computeLikelihoodRatioCost(u, theta_d, t, distribution_idx, lambda, alpha);
    }
    __syncthreads();
  }

  // Add all costs together
  int prev_size = BLOCKSIZE_X;
  running_cost = &running_cost_shared[blockDim.x * thread_idz];
#if false
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
  { // unroll the last warp
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
  computeAndSaveCost(num_rollouts, num_timesteps, global_idx, costs, y, running_cost[0] / (num_timesteps - 1), theta_c,
                     trajectory_costs_d);
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

template <class DYN_T, class COST_T, class SAMPLING_T>
void launchFastRolloutKernel(DYN_T* dynamics, COST_T* costs, SAMPLING_T* sampling, float& dt, const int& num_timesteps,
                             const int& num_rollouts, const int& optimization_stride, float& lambda, float& alpha,
                             float* __restrict__ init_x_d, float* __restrict__ y_d,
                             float* __restrict__ trajectory_costs, dim3& dimDynBlock, dim3& dimCostBlock,
                             cudaStream_t stream, bool synchronize)
{
  // Run Dynamics
  const int gridsize_x = mppi::math::int_ceil(num_rollouts, dimDynBlock.x);
  dim3 dimGrid(gridsize_x, 1, 1);
  const int dynamics_num_shared = dimDynBlock.x * dimDynBlock.z;
  unsigned dynamics_shared_size =
      sizeof(float) * (3 * mppi::math::nearest_quotient_4(dynamics_num_shared * DYN_T::STATE_DIM) +
                       mppi::math::nearest_quotient_4(dynamics_num_shared * DYN_T::OUTPUT_DIM) +
                       mppi::math::nearest_quotient_4(dynamics_num_shared * DYN_T::CONTROL_DIM)) +
      mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      mppi::math::int_ceil(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * mppi::math::int_ceil(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  rolloutDynamicsKernel<DYN_T, SAMPLING_T><<<dimGrid, dimDynBlock, dynamics_shared_size, stream>>>(
      dynamics, sampling, dt, num_timesteps, optimization_stride, num_rollouts, init_x_d, y_d);

  // Run Costs
  dim3 dimCostGrid(num_rollouts, 1, 1);
  const int cost_num_shared = dimCostBlock.x * dimCostBlock.z;
  unsigned cost_shared_size =
      sizeof(float) * (mppi::math::nearest_quotient_4(cost_num_shared * COST_T::OUTPUT_DIM) +
                       mppi::math::nearest_quotient_4(cost_num_shared * COST_T::CONTROL_DIM) +
                       mppi::math::nearest_quotient_4(cost_num_shared)) +
      sizeof(int) * mppi::math::nearest_quotient_4(cost_num_shared) +
      mppi::math::int_ceil(COST_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * mppi::math::int_ceil(COST_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4)) +
      mppi::math::int_ceil(SAMPLING_T::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
      dynamics_num_shared * mppi::math::int_ceil(SAMPLING_T::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4));

  rolloutCostKernel<DYN_T, COST_T, NUM_ROLLOUTS, COST_BLOCK_X><<<dimCostGrid, dimCostBlock, cost_shared_size, stream>>>(
      costs, sampling, dt, num_timesteps, num_rollouts, lambda, alpha, init_x_d, y_d, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}
