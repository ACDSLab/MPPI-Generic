#include <mppi/core/mppi_common.cuh>
#include <curand.h>
#include <mppi/utils/gpu_err_chk.cuh>

namespace mppi_common {
  /*******************************************************************************************************************
  * Kernel Functions
  *******************************************************************************************************************/
  // TODO remove dt
  template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y,
         int NUM_ROLLOUTS, int BLOCKSIZE_Z>
  __global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs,
                               float dt,
                               int num_timesteps,
                               float* x_d,
                               float* u_d,
                               float* du_d,
                               float* sigma_u_d,
                               float* trajectory_costs_d) {
    // Get thread and block id
    int thread_idx = threadIdx.x;
    int thread_idy = threadIdx.y;
    int thread_idz = threadIdx.z;
    int block_idx = blockIdx.x;
    int global_idx = BLOCKSIZE_X * block_idx + thread_idx;

    // Create shared state and control arrays
    __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
    __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
    __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
    __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
    __shared__ float sigma_u[DYN_T::CONTROL_DIM];

    // Create a shared array for the dynamics model to use
    __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X];

    // Create local state, state dot and controls
    float* x;
    float* xdot;
    float* u;
    float* du;
    // float* sigma_u;

    //Initialize running cost and total cost
    float running_cost = 0;
    //Load global array to shared array
    if (global_idx < NUM_ROLLOUTS) {
      x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
      xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
      u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
      du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
      // sigma_u = &sigma_u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    }
    __syncthreads();
    loadGlobalToShared(DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, NUM_ROLLOUTS,
                       BLOCKSIZE_Y, global_idx, thread_idy,
                       thread_idz, x_d, sigma_u_d, x, xdot, u, du, sigma_u);
    __syncthreads();


    if (global_idx < NUM_ROLLOUTS) {
      /*<----Start of simulation loop-----> */
      for (int t = 0; t < num_timesteps; t++) {
        //Load noise trajectories scaled by the exploration factor
        injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps,
                          t, global_idx, thread_idy, u_d, du_d, sigma_u, u, du);
        __syncthreads();

        // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
        // usually just control clamping
        // Clamp the control in both the importance sampling sequence and the disturbed sequence. TODO remove extraneous call?
        dynamics->enforceConstraints(x, &du_d[global_idx*num_timesteps*DYN_T::CONTROL_DIM + t*DYN_T::CONTROL_DIM]);
        dynamics->enforceConstraints(x, u);

        __syncthreads();

        //Accumulate running cost
        running_cost += costs->computeRunningCost(x, u, du, sigma_u, t)*dt;
        __syncthreads();

        //Compute state derivatives
        dynamics->computeStateDeriv(x, u, xdot, theta_s);
        __syncthreads();

        //Increment states
        dynamics->updateState(x, xdot, dt);
        __syncthreads();
      }
      //Compute terminal cost and the final cost for each thread
      computeAndSaveCost(NUM_ROLLOUTS, global_idx, costs, x, running_cost,
                        trajectory_costs_d + thread_idz * NUM_ROLLOUTS);
    }

    __syncthreads();
  }

  __global__ void normExpKernel(int num_rollouts,
                                float* trajectory_costs_d,
                                float gamma,
                                float baseline) {
    int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * blockDim.z + \
      threadIdx.z;

    if (global_idx < num_rollouts * blockDim.z) {
      float cost_dif = trajectory_costs_d[global_idx] - baseline;
      trajectory_costs_d[global_idx] = expf(-gamma*cost_dif);
    }
  }

  template<int CONTROL_DIM, int NUM_ROLLOUTS, int SUM_STRIDE>
  __global__ void weightedReductionKernel(float*  exp_costs_d,
                                          float* du_d,
                                          float* du_new_d,
                                          float normalizer,
                                          int num_timesteps) {
    int thread_idx = threadIdx.x;  // Rollout index
    int block_idx = blockIdx.x; // Timestep

    //Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
    __shared__ float u_intermediate[CONTROL_DIM * ((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1)];

    float u[CONTROL_DIM];
    setInitialControlToZero(CONTROL_DIM, thread_idx, u, u_intermediate);

    __syncthreads();

    //Sum the weighted control variations at a desired stride
    strideControlWeightReduction(NUM_ROLLOUTS, num_timesteps, SUM_STRIDE,
                                 thread_idx,
                                 block_idx, CONTROL_DIM,
                                 exp_costs_d, normalizer,
                                 du_d, u, u_intermediate);

    __syncthreads();

    //Sum all weighted control variations
    rolloutWeightReductionAndSaveControl(thread_idx, block_idx,
                                         NUM_ROLLOUTS, num_timesteps,
                                         CONTROL_DIM, SUM_STRIDE,
                                         u, u_intermediate, du_new_d);

    __syncthreads();
  }


    /*******************************************************************************************************************
     * Rollout Kernel Helpers
    *******************************************************************************************************************/
    __device__ void loadGlobalToShared(int state_dim, int control_dim,
                                       int num_rollouts, int blocksize_y,
                                       int global_idx, int thread_idy,
                                       int thread_idz,
                                       const float* x_device,
                                       const float* sigma_u_device,
                                       float* x_thread,
                                       float* xdot_thread,
                                       float* u_thread,
                                       float* du_thread,
                                       float* sigma_u_thread) {
      //Transfer to shared memory
      int i;
      if (global_idx < num_rollouts) {
        for (i = thread_idy; i < state_dim; i += blocksize_y) {
          x_thread[i] = x_device[i + state_dim * thread_idz];
          xdot_thread[i] = 0;
        }
        for (i = thread_idy; i < control_dim; i += blocksize_y) {
          u_thread[i] = 0;
          du_thread[i] = 0;
          // Only do in threadIdx.x and parallelize along threadIdx.y
          // sigma_u_thread[i] = sigma_u_device[i];
        }
      }
      if (threadIdx.x == 0 /*&& threadIdx.z == 0*/) {
        for(i = thread_idy; i < control_dim; i +=blocksize_y){
          sigma_u_thread[i] = sigma_u_device[i];
        }
      }
      // for (i = blockDim.y*blockDim.x*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x; i < control_dim; i+= blockDim.z*blockDim.x*blockDim.y){

      // }
    }

    __device__ void injectControlNoise(int control_dim,
                                       int blocksize_y, int num_rollouts,
                                       int num_timesteps,
                                       int current_timestep,
                                       int global_idx,
                                       int thread_idy,
                                       const float* u_traj_device,
                                       float* ep_v_device,
                                       const float* sigma_u_thread,
                                       float* u_thread, float* du_thread) {
        int control_index = global_idx*control_dim*num_timesteps + current_timestep * control_dim;
        //Load the noise trajectory scaled by the exploration factor
        // The prior loop already guarantees that the global index is less than the number of rollouts
        for (int i = thread_idy; i < control_dim; i += blocksize_y) {
            //Keep one noise free trajectory
            if (global_idx == 0){
                du_thread[i] = 0;
                u_thread[i] = u_traj_device[current_timestep * control_dim + i];
            }
            //Generate 1% zero control trajectory
            else if (global_idx >= 0.99*num_rollouts) {
                du_thread[i] = ep_v_device[control_index + i] * sigma_u_thread[i];
                u_thread[i] = du_thread[i];
            }
            else {
                du_thread[i] = ep_v_device[control_index + i] * sigma_u_thread[i];
                u_thread[i] = u_traj_device[current_timestep * control_dim + i] + du_thread[i];
            }
            // Saves the control but doesn't clamp it.
            ep_v_device[control_index + i] = u_thread[i];
        }
    }

    template<class COST_T>
    __device__ void computeAndSaveCost(int num_rollouts, int global_idx, COST_T* costs, float* x_thread,
                                        float running_cost, float* cost_rollouts_device) {
        if (global_idx < num_rollouts) {
            cost_rollouts_device[global_idx] = running_cost + costs->terminalCost(x_thread);
        }
    }

    /*******************************************************************************************************************
     * NormExp Kernel Helpers
    *******************************************************************************************************************/
    float computeBaselineCost(float* cost_rollouts_host, int num_rollouts) { // TODO if we use standard containers in MPPI, should this be replaced with a min algorithm?
        float baseline = cost_rollouts_host[0];
        // Find the minimum cost trajectory
        for (int i = 0; i < num_rollouts; ++i) {
            if (cost_rollouts_host[i] < baseline) {
                baseline = cost_rollouts_host[i];
            }
        }
        return baseline;
    }

    float computeNormalizer(float* cost_rollouts_host, int num_rollouts) {
        float normalizer = 0.f;
        for (int i = 0; i < num_rollouts; ++i) {
            normalizer += cost_rollouts_host[i];
        }
        return normalizer;
    }

    /*******************************************************************************************************************
     * Weighted Reduction Kernel Helpers
    *******************************************************************************************************************/
    __device__ void setInitialControlToZero(int control_dim, int thread_idx, float* u, float* u_intermediate) {
        // TODO replace with memset?
        for (int i = 0; i < control_dim; i++) {
            u[i] = 0;
            u_intermediate[thread_idx * control_dim + i] = 0;
        }
    }

    __device__ void strideControlWeightReduction(int num_rollouts,
                                                 int num_timesteps,
                                                 int sum_stride,
                                                 int thread_idx,
                                                 int block_idx,
                                                 int control_dim,
                                                 float* exp_costs_d,
                                                 float normalizer,
                                                 float* du_d,
                                                 float* u,
                                                 float* u_intermediate) {
        // int index = thread_idx * sum_stride + i;
        for (int i = 0; i < sum_stride; ++i) { // Iterate through the size of the subsection
            if ((thread_idx * sum_stride + i) < num_rollouts) { //Ensure we do not go out of bounds
                float weight = exp_costs_d[thread_idx * sum_stride + i] / normalizer; // compute the importance sampling weight
                for (int j = 0; j < control_dim; ++j) { // Iterate through the control dimensions
                    // Rollout index: (thread_idx*sum_stride + i)*(num_timesteps*control_dim)
                    // Current timestep: block_idx*control_dim
                    u[j] = du_d[(thread_idx * sum_stride + i)*(num_timesteps*control_dim) + block_idx*control_dim + j];
                    u_intermediate[thread_idx * control_dim + j] += weight * u[j];
                }
            }
        }
    }

    __device__ void rolloutWeightReductionAndSaveControl(int thread_idx, int block_idx, int num_rollouts, int num_timesteps,
            int control_dim, int sum_stride, float* u, float* u_intermediate, float* du_new_d) {
        if (thread_idx == 0 && block_idx < num_timesteps) { //block index refers to the current timestep
            for (int i = 0; i < control_dim; ++i) { // TODO replace with memset?
                u[i] = 0;
            }
            for (int i = 0; i < ((num_rollouts - 1) / sum_stride + 1); ++i) { // iterate through the each subsection
                for (int j = 0; j < control_dim; ++j) {
                    u[j] += u_intermediate[i * control_dim + j];
                }
            }
            for (int i = 0; i < control_dim; i++) {
                du_new_d[block_idx * control_dim + i] = u[i];
            }
        }
    }

    /*******************************************************************************************************************
     * Launch Functions
    *******************************************************************************************************************/
    template<class DYN_T, class COST_T, int NUM_ROLLOUTS, int BLOCKSIZE_X,
             int BLOCKSIZE_Y, int BLOCKSIZE_Z = 1>
    void launchRolloutKernel(DYN_T* dynamics, COST_T* costs, float dt,
                             int num_timesteps, float* x_d, float* u_d,
                             float* du_d, float* sigma_u_d,
                             float* trajectory_costs, cudaStream_t stream) {
      const int gridsize_x = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;
      dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
      dim3 dimGrid(gridsize_x, 1, 1);
      rolloutKernel<DYN_T, COST_T, BLOCKSIZE_X, BLOCKSIZE_Y, NUM_ROLLOUTS, BLOCKSIZE_Z><<<dimGrid, dimBlock, 0, stream>>>(dynamics, costs, dt,
              num_timesteps, x_d, u_d, du_d, sigma_u_d, trajectory_costs);
      CudaCheckError();
      HANDLE_ERROR( cudaStreamSynchronize(stream) );
    }

    void launchNormExpKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float gamma, float baseline, cudaStream_t stream) {
        dim3 dimBlock(blocksize_x, 1, 1);
        dim3 dimGrid((num_rollouts - 1) / blocksize_x + 1, 1, 1);
        normExpKernel<<<dimGrid, dimBlock, 0, stream>>>(num_rollouts, trajectory_costs_d, gamma, baseline);
        CudaCheckError();
        HANDLE_ERROR( cudaStreamSynchronize(stream) );
    }

    template<class DYN_T, int NUM_ROLLOUTS, int SUM_STRIDE >
    void launchWeightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d, float normalizer,
            int num_timesteps, cudaStream_t stream) {
        dim3 dimBlock((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1, 1, 1);
        dim3 dimGrid(num_timesteps, 1, 1);
        weightedReductionKernel<DYN_T::CONTROL_DIM, NUM_ROLLOUTS, SUM_STRIDE><<<dimGrid, dimBlock, 0, stream>>>
                (exp_costs_d, du_d, du_new_d, normalizer, num_timesteps);
        CudaCheckError();
        HANDLE_ERROR( cudaStreamSynchronize(stream) );
    }

}

namespace rmppi_kernels {
  template <class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y>
  __global__ void initEvalKernel(DYN_T* dynamics,
                                 COST_T* costs,
                                 int samples_per_condition,
                                 int num_timesteps,
                                 int ctrl_stride,
                                 float dt,
                                 int* strides_d,
                                 float* exploration_var_d,
                                 float* states_d,
                                 float* control_d,
                                 float* control_noise_d,
                                 float* costs_d) {
    int i,j;
    int tdx = threadIdx.x;
    int tdy = threadIdx.y;
    int bdx = blockIdx.x;

    //Initialize the local state, controls, and noise
    float* state;
    float* state_der;
    float* control;
    float* control_noise;  // du
    float* exploration_var;  //nu


    //Create shared arrays for holding state and control data.
    __shared__ float state_shared[BLOCKSIZE_X*DYN_T::STATE_DIM];
    __shared__ float state_der_shared[BLOCKSIZE_X*DYN_T::STATE_DIM];
    __shared__ float control_shared[BLOCKSIZE_X*DYN_T::CONTROL_DIM];
    __shared__ float control_noise_shared[BLOCKSIZE_X*DYN_T::CONTROL_DIM];
    __shared__ float exploration_variance[BLOCKSIZE_X*DYN_T::CONTROL_DIM]; // Each thread has its own copy

    //Create a shared array for the dynamics model to use
    __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X];


    float running_cost = 0;  //Initialize trajectory cost

    int global_idx = BLOCKSIZE_X*bdx + tdx;  // Set the global index for CUDA threads
    int condition_idx = global_idx / samples_per_condition; // Set the index for our candidate
    int stride = strides_d[condition_idx];  // Each candidate can have a different starting stride

    // Get the pointer that belongs to the current thread with respect to the shared arrays
    state = &state_shared[tdx*DYN_T::STATE_DIM];
    state_der = &state_der_shared[tdx*DYN_T::STATE_DIM];
    control = &control[tdx*DYN_T::CONTROL_DIM];
    control_noise = &control_noise_shared[tdx*DYN_T::CONTROL_DIM];
    exploration_var = &exploration_variance[tdx*DYN_T::CONTROL_DIM];

    // Copy the state to the thread
    for (i = tdy; i < DYN_T::STATE_DIM; i+= blockDim.y) {
      state[i] = states_d[condition_idx*DYN_T::STATE_DIM + i]; // states_d holds each condition
    }

    // Copy the exploration noise to the thread
    for (i = tdy; i < DYN_T::CONTROL_DIM; i += blockDim.y) {
      control[i] = 0;
      control_noise[i] = 0;
      exploration_var[i] = exploration_var_d[i];
    }

    __syncthreads();

    for (i = 0; i < num_timesteps; ++i) { // Outer loop iterates on timesteps
      // Inject the control noise
      for (j = tdy; j < DYN_T::CONTROL_DIM; j += blockDim.y) {
        if (i + stride >= num_timesteps) {  // Pad the end of the controls with the last control
          control[j] = control_d[num_timesteps*DYN_T::CONTROL_DIM + j];
        } else {
          control[j] = control_d[(i + stride)*DYN_T::CONTROL_DIM + j];
        }

        // First rollout is noise free
        if (global_idx % samples_per_condition == 0 || i < ctrl_stride) {
          control_noise[j] = 0.0;
        } else {
          control_noise[j] = control_noise_d[num_timesteps*DYN_T::CONTROL_DIM*global_idx +
                                             i*DYN_T::CONTROL_DIM + j]*exploration_var[j];
        }

        // Sum the control and the noise
        control[j] += control_noise[j];
      } // End inject control noise

      __syncthreads();
      if (tdy == 0) {
        dynamics->enforceConstraints(state, &control_noise_d[num_timesteps*DYN_T::CONTROL_DIM*global_idx +
                                                             i*DYN_T::CONTROL_DIM]);
        dynamics->enforceConstraints(state, control);
      }

      __syncthreads();
      if (tdy == 0) { // Only compute once per global index.
        running_cost +=
                (costs->computeCost(state, control, control_noise, exploration_var, i) * dt - running_cost) / (1.0 * i);
      }
      __syncthreads();

      //Compute state derivatives
      dynamics->computeStateDeriv(state, control, state_der, theta_s);
      __syncthreads();

      //Increment states
      dynamics->updateState(state, state_der, dt);
      __syncthreads();
      }
    // End loop outer loop on timesteps

    if (tdy == 0) {  // Only save the costs once per global idx (thread y is only for parallelization)
      costs_d[global_idx] = running_cost; // This is the running average of the costs along the trajectory
    }
  }

  template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y>
  void launchInitEvalKernel(DYN_T* dynamics,
                            COST_T* costs,
                            int samples_per_condition,
                            int num_candidates,
                            int num_timesteps,
                            int ctrl_stride,
                            float dt,
                            int* strides_d,
                            float* exploration_var_d,
                            float* states_d,
                            float* control_d,
                            float* control_noise_d,
                            float* costs_d) {

    int GRIDSIZE_X = num_candidates * samples_per_condition / BLOCKSIZE_X;
    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 dimGrid(GRIDSIZE_X, 1, 1);
    initEvalKernel<DYN_T, COST_T, BLOCKSIZE_X, BLOCKSIZE_Y><<<dimGrid, dimBlock, 0>>>(dynamics, costs,
            samples_per_condition, num_timesteps, ctrl_stride, dt, strides_d, exploration_var_d, states_d,
            control_d, control_noise_d, costs_d);

  }

  // Newly Written
  template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y,
         int NUM_ROLLOUTS, int BLOCKSIZE_Z>
  __global__ void RMPPIRolloutKernel(DYN_T * dynamics, COST_T* costs,
                                     float dt,
                                     int num_timesteps,
                                     float* x_d,
                                     float* u_d,
                                     float* du_d,
                                     float* feedback_gains_d,
                                     float* sigma_u_d,
                                     float* trajectory_costs_d,
                                     float lambda) {
    int thread_idx = threadIdx.x;
    int thread_idy = threadIdx.y;
    int thread_idz = threadIdx.z;
    int block_idx = blockIdx.x;
    int global_idx = BLOCKSIZE_X * block_idx + thread_idx;

    // Create shared memory for state and control
    __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
    __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
    __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
    __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
    __shared__ float sigma_u[DYN_T::CONTROL_DIM];

    // Create a shared array for the nominal costs calculations
    __shared__ float running_state_cost_nom_shared[BLOCKSIZE_X];

    // Create a shared array for the dynamics model to use
    __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X];

    // Create local state, state dot and controls
    float* x;
    float* x_other;
    float* xdot;
    float* u;
    float* du;
    float* fb_gain;
    // The array to hold K(x,x*)
    float* fb_control[DYN_T::CONTROL_DIM];

    int t = 0;
    int i = 0;
    int j = 0;

    // Initialize running costs
    float running_cost_real = 0;
    float* running_state_cost_nom;
    float running_tracking_cost_real = 0;

    // Load global array into shared memory
    if (global_idx < NUM_ROLLOUTS) {
      // Actual or nominal
      x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
      // The opposite state from above
      x_other = &x_shared[(blockDim.x * (1 - thread_idz) + thread_idx) * DYN_T::STATE_DIM];
      xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
      // Base trajectory
      u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
      // Noise added to trajectory
      du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
      // Nominal State Cost
      running_state_cost_nom = &running_state_cost_nom_shared[thread_idx];
    }

    *running_state_cost_nom = 0;

    __syncthreads();
    // Load memory into appropriate arrays
    loadGlobalToShared(DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, NUM_ROLLOUTS,
                      BLOCKSIZE_Y, global_idx, thread_idy,
                      thread_idz, x_d, sigma_u_d, x, xdot, u, du, sigma_u);
    __syncthreads();
    //TODO: Need to load feedback gains as well
    for (t = 0; t < num_timesteps; t++) {
      injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps,
                        t, global_idx, thread_idy, u_d, du_d, sigma_u, u, du);
      // Now find feedback control
      float e;
      // Feedback gains at time t
      fb_gain = &feedback_gains_d[t * DYN_T::CONTROL_DIM * DYN_T::STATE_DIM];
      for (i = 0; i < DYN_T::CONTROL_DIM; i++) {
        fb_control[i] = 0;
      }
      // Don't enter for loop if in nominal states (thread_idz == 1)
      for (i = 0; i < DYN_T::STATE_DIM * (1 - thread_idz); i++) {
        // Find difference between nominal and actual
        e = (x - x_other);
        for (j = 0; j < DYN_T::CONTROL_DIM; j++) {
          // Assuming column major storage atm. TODO Double check storage option
          fb_control[j] += fb_gain[i * DYN_T::CONTROL_DIM + j] * e;
        }
      }
      for (i = 0; i < DYN_T::CONTROL_DIM; i++) {
        u[i] += fb_control[i];
      }

      __syncthreads();
      // Clamp the control in both the importance sampling sequence and the disturbed sequence. TODO remove extraneous call?
      dynamics->enforceConstraints(x, du);
      dynamics->enforceConstraints(x, u);

      __syncthreads();
      // Calculate All the costs
      float curr_state_cost =  costs->computeStateCost(x);

      // Nominal system is where thread_idz == 1
      if (thread_idz == 1) {
        *running_state_cost_nom += curr_state_cost;
      }
      // Real system cost update when thread_idz == 0
      if (thread_idz == 0) {
        running_cost_real += (curr_state_cost +
          costs->computeLikelihoodRatioCost(u, du, sigma_u, lambda));

        running_tracking_cost_real += (curr_state_cost +
          costs->computeFeedbackCost(fb_control, sigma_u, lambda));
      }

      // Non if statement version
      // running_cost_real += (1 - thread_idz) * (curr_state_cost +
      //   costs->computeLikelihoodRatioCost(u, du, sigma_u, t));
      // running_tracking_cost_real += (1 - thread_idz) * (curr_state_cost +
      //   costs->computeFeedbackCost(fb_control, sigma_u));

      __syncthreads();
      // dynamics update
      dynamics->computeStateDeriv(x, u, xdot, theta_s);
      __syncthreads();
      dynamics->updateState(x, xdot, dt);
      __syncthreads();
    }
    // Choose which cost to use for nominal cost
    // TODO: Replace with parameter passed in
    float value_func_threshold_ = 10;
    /** TODO: This will not work in current setup because running_state_cost_nom
    * and running_tracking_cost_real are calculated by different threads (thread_z 1 or 0)
    * Need to create shared memory for some parts of it
    **/
    float running_cost_nom  = 0;
    if (thread_idz == 0) {
      running_cost_nom = 0.5 * (*running_state_cost_nom) + 0.5 *
        fmaxf(fminf(running_tracking_cost_real, value_func_threshold_), *running_state_cost_nom);

      for(t = 0; t < num_timesteps - 1; t++) {
        // Get u(t) and noise at time t
        injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps,
          t, global_idx, thread_idy, u_d, du_d, sigma_u, u, du);
        __syncthreads();
        running_cost_nom += costs->computeLikelihoodRatioCost(u, du, sigma_u, lambda);
      }
    }
    __syncthreads();
    // Copy costs over to correct locations
    /** TODO: Right now copying can only occur from the real system threads
    * We can leave it as is or we could copy the nominal trajectory costs
    * into a shared memory location so that we could try to use computeAndSaveCost
    */
    if (thread_idz == 0) {
      // Only the threadds running the actual system have the final running costs for both
      // real and nominal
      if (global_idx < NUM_ROLLOUTS) {
        // Actual System cost
        trajectory_costs_d[global_idx] = running_cost_real;
        // Nominal System Cost - Again this is actaully only  known on real system threads
        trajectory_costs_d[global_idx + NUM_ROLLOUTS] = running_cost_nom;
      }
    }
    __syncthreads();
  }

  /*******************************************************************************************************************
   * Launch Functions
   *******************************************************************************************************************/
  template<class DYN_T, class COST_T, int NUM_ROLLOUTS, int BLOCKSIZE_X,
            int BLOCKSIZE_Y, int BLOCKSIZE_Z>
  void launchRMPPIRolloutKernel(DYN_T* dynamics, COST_T* costs,
                                float dt,
                                int num_timesteps,
                                float* x_d,
                                float* u_d,
                                float* du_d,
                                float* feedback_gains_d,
                                float* sigma_u_d,
                                float* trajectory_costs,
                                float lambda,
                                cudaStream_t stream) {
    const int gridsize_x = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;
    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    dim3 dimGrid(gridsize_x, 1, 1);
    RMPPIRolloutKernel<DYN_T, COST_T, BLOCKSIZE_X, BLOCKSIZE_Y, NUM_ROLLOUTS,
                      BLOCKSIZE_Z><<<dimGrid, dimBlock, 0, stream>>>(
                        dynamics, costs, dt, num_timesteps, x_d, u_d, du_d,
                        feedback_gains_d, sigma_u_d, trajectory_costs, lambda);
    CudaCheckError();
    HANDLE_ERROR( cudaStreamSynchronize(stream) );
  }

}
