#include <mppi/utils/math_utils.h>

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
                                             const float* __restritct__ du_d, float* __restrict__ u,
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

template <class DYN_T, class SAMPLING_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int NUM_ROLLOUTS, int BLOCKSIZE_Z>
__global__ void rolloutDynamicsKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling, float dt,
                                      int num_timesteps, int optimization_stride, const float* __restrict__ init_x_d,
                                      float* __restrict__ y_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int block_idx = blockIdx.x;
  const int global_idx = BLOCKSIZE_X * block_idx + thread_idx;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int distribution_idx = threadIdx.z;
  const int distribution_dim = blockDim.z;
  const int sample_dim = blockDim.x;

  extern __shared__ float entire_buffer[];

  float* x_shared = entire_buffer;
  float* x_next_shared = &x_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[mppi::math::nearest_quotient_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD and SHARED_MEM_REQUEST_BLK portions to be aligned
  // to the float4 boundary.
  const int size_of_theta_s_bytes =
      mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_GRD, sizeof(float4)) +
      sample_dim * distribution_dim * mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_BLK, sizeof(float4));
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];

  // Create shared state and control arrays
  // __shared__ float4 x_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z, 4)];
  // __shared__ float4 x_next_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z, 4)];
  // __shared__ float4 y_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::OUTPUT_DIM * BLOCKSIZE_Z, 4)];
  // __shared__ float4 xdot_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z, 4)];
  // __shared__ float4 u_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z, 4)];
  // __shared__ float4 du_shared[mppi::math::int_ceil(BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z, 4)];
  // __shared__ float4 sigma_u[mppi::math::int_ceil(DYN_T::CONTROL_DIM, 4)];

  // // Create a shared array for the dynamics model to use
  // __shared__ float4 theta_s4[mppi::math::int_ceil(DYN_T::SHARED_MEM_REQUEST_GRD / sizeof(float) + 1 +
  //                                                     DYN_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X * BLOCKSIZE_Z,
  //                                                 4)];
  // float* theta_s = reinterpret_cast<float*>(theta_s4);

  // Create local state, state dot and controls
  float* x;
  float* x_next;
  float* x_temp;
  float* xdot;
  float* u;
  // float* du;
  float* y;

  // Load global array to shared array
  if (global_idx < NUM_ROLLOUTS)
  {
    x = &(reinterpret_cast<float*>(x_shared)[shared_idx * DYN_T::STATE_DIM]);
    x_next = &(reinterpret_cast<float*>(x_next_shared)[shared_idx * DYN_T::STATE_DIM]);
    y = &(reinterpret_cast<float*>(y_shared)[shared_idx * DYN_T::OUTPUT_DIM]);
    xdot = &(reinterpret_cast<float*>(xdot_shared)[shared_idx * DYN_T::STATE_DIM]);
    u = &(reinterpret_cast<float*>(u_shared)[shared_idx * DYN_T::CONTROL_DIM]);
    // du = &(reinterpret_cast<float*>(du_shared)[shared_idx * DYN_T::CONTROL_DIM]);
  }
  // TODO: Replace with new loadGlobaToShared that doesn't reuire sigma or du
  // loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(NUM_ROLLOUTS, BLOCKSIZE_Y, global_idx, thread_idy,
  //                                                          thread_idz, init_x_d, sigma_u_d, x, xdot, u, du,
  //                                                          reinterpret_cast<float*>(sigma_u));
  __syncthreads();

  if (global_idx < NUM_ROLLOUTS)
  {
    /*<----Start of simulation loop-----> */
    dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0, dt);
    sampling->initializeDistributions(s, 0.0, dt, theta_d_shared);
    for (int t = 0; t < num_timesteps; t++)
    {
      // Load noise trajectories scaled by the exploration factor
      // injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps, t, global_idx, thread_idy,
      //                    optimization_stride, u_d, du_d, reinterpret_cast<float*>(sigma_u), u, du);
      sampling->readControlSample(global_idx, t, distribution_idx, s, u, theta_d_shared, blockDim.y, thread_idy);
      // du_d is now v
      __syncthreads();

      // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
      // usually just control clamping
      // calls enforceConstraints on both since one is used later on in kernel (u), du_d is what is sent back to the CPU

      // TODO: Replace with enforcing constraints method from sampling distribution
      // dynamics->enforceConstraints(x, &du_d[(NUM_ROLLOUTS * num_timesteps * threadIdx.z +  // z part
      //                                        global_idx * num_timesteps + t) *
      //                                       DYN_T::CONTROL_DIM]);
      dynamics->enforceConstraints(x, u);
      __syncthreads();

      // Increment states
      dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
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
