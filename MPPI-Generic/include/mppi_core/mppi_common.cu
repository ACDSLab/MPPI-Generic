#include "mppi_core/mppi_common.cuh"
#include <curand.h>
#include "utils/gpu_err_chk.cuh"

namespace mppi_common {
    /*******************************************************************************************************************
     * Kernel Functions
    *******************************************************************************************************************/
    template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int NUM_ROLLOUTS>
     __global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs, float dt,
                                    int num_timesteps, float* x_d, float* u_d, float* du_d, float* sigma_u_d,
                                    float* trajectory_costs_d) {
        //Get thread and block id
        int thread_idx = threadIdx.x;
        int thread_idy = threadIdx.y;
        int block_idx = blockIdx.x;
        int global_idx = BLOCKSIZE_X * block_idx + thread_idx;

        //Create shared state and control arrays
        __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM];
        __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM];
        __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM];
        __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM];
        __shared__ float sigma_u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM];

        //Create local state, state dot and controls
        float* x;
        float* xdot;
        float* u;
        float* du;
        float* sigma_u;

        //Initialize running cost and total cost
        float running_cost = 0;

        //Load global array to shared array
        if (global_idx < NUM_ROLLOUTS) {
            x = &x_shared[thread_idx * DYN_T::STATE_DIM];
            xdot = &xdot_shared[thread_idx * DYN_T::STATE_DIM];
            u = &u_shared[thread_idx * DYN_T::CONTROL_DIM];
            du = &du_shared[thread_idx * DYN_T::CONTROL_DIM];
            sigma_u = &sigma_u_shared[thread_idx * DYN_T::CONTROL_DIM];
        }
        __syncthreads();
        loadGlobalToShared(DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, NUM_ROLLOUTS, BLOCKSIZE_Y,global_idx, thread_idy, x_d, sigma_u_d,
                            x, xdot, u, du, sigma_u);
        __syncthreads();

        /*<----Start of simulation loop-----> */
        for (int t = 0; t < num_timesteps; t++) {
            if (global_idx < NUM_ROLLOUTS) {
                //Load noise trajectories scaled by the exploration factor
                injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps,
                                   t, global_idx, thread_idy, u_d, du_d, sigma_u, u, du);
                __syncthreads();

                //Accumulate running cost
                computeRunningCostAllRollouts(costs, dt, x, u, du, sigma_u, running_cost);
                __syncthreads();

                //Compute state derivatives
                computeStateDerivAllRollouts(dynamics, x, u, xdot);
                __syncthreads();

                //Increment states
                incrementStateAllRollouts(DYN_T::STATE_DIM, BLOCKSIZE_Y, thread_idy, dt, x, xdot);
                __syncthreads();
            }
        }

        //Compute terminal cost and the final cost for each thread
        computeAndSaveCost(global_idx, costs, x, running_cost, trajectory_costs_d);
        __syncthreads();
    }

    __global__ void normExpKernel(int num_rollouts, float* trajectory_costs_d, float gamma, float baseline) {
        int global_idx = blockDim.x * blockIdx.x +  threadIdx.x;

        if (global_idx < num_rollouts) {
            float cost_dif = trajectory_costs_d[global_idx] - baseline;
            trajectory_costs_d[global_idx] = expf(-gamma*cost_dif);
        }
    }

    template<int CONTROL_DIM, int NUM_ROLLOUTS, int SUM_STRIDE>
    __global__ void weightedReductionKernel(float*  exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d,
            float normalizer, int num_timesteps) {
        int thread_idx = threadIdx.x;  // Rollout index
        int block_idx = blockIdx.x; // Timestep

        //Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
        __shared__ float u_intermediate[CONTROL_DIM * ((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1)];

        float u[CONTROL_DIM];
        setInitialControlToZero(CONTROL_DIM, thread_idx, u, u_intermediate);

        __syncthreads();

        //Sum the weighted control variations at a desired stride
        strideControlWeightReduction(NUM_ROLLOUTS, num_timesteps, SUM_STRIDE, thread_idx, block_idx, CONTROL_DIM,
                exp_costs_d, normalizer, du_d, sigma_u_d, u, u_intermediate);

        __syncthreads();

        //Sum all weighted control variations
        rolloutWeightReductionAndSaveControl(thread_idx, block_idx, NUM_ROLLOUTS, num_timesteps, CONTROL_DIM,
                SUM_STRIDE, u, u_intermediate, du_new_d);
    }


    /*******************************************************************************************************************
     * Rollout Kernel Helpers
    *******************************************************************************************************************/
    __device__ void loadGlobalToShared(int state_dim, int control_dim, int num_rollouts, int blocksize_y, int global_idx, int thread_idy,
                                        const float* x_device, const float* sigma_u_device, float* x_thread,
                                        float* xdot_thread, float* u_thread, float* du_thread, float* sigma_u_thread) {
        //Transfer to shared memory
        int i;
        if (global_idx < num_rollouts) {
            for (i = thread_idy; i < state_dim; i += blocksize_y) {
                x_thread[i] = x_device[i];
                xdot_thread[i] = 0;
            }
            for (i = thread_idy; i < control_dim; i += blocksize_y) {
                u_thread[i] = 0;
                du_thread[i] = 0;
                sigma_u_thread[i] = sigma_u_device[i];
            }
        }
    }

    __device__ void injectControlNoise(int control_dim, int blocksize_y, int num_rollouts, int num_timesteps,
            int current_timestep, int global_idx, int thread_idy,
            const float* u_traj_device, float* ep_v_device, const float* sigma_u_thread,
            float* u_thread, float* du_thread) {
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
                du_thread[i] = ep_v_device[global_idx*control_dim*num_timesteps + current_timestep * control_dim + i] * sigma_u_thread[i];
                u_thread[i] = du_thread[i];
            }
            else {
                du_thread[i] = ep_v_device[global_idx*control_dim*num_timesteps + current_timestep * control_dim + i] * sigma_u_thread[i];
                u_thread[i] = u_traj_device[current_timestep * control_dim + i] + du_thread[i];
            }
            ep_v_device[control_dim*num_timesteps*global_idx + control_dim*current_timestep + i] = u_thread[i];
        }
    }

    template<class COST_T>
    __device__ void computeRunningCostAllRollouts(COST_T* costs, float dt, float* x_thread, float* u_thread, float* du_thread, float* sigma_thread, float& running_cost) {
        // The prior loop already guarantees that the global index is less than the number of rollouts
        running_cost += costs->computeRunningCost(x_thread, u_thread, du_thread, sigma_thread)*dt;
    }

    template<class DYN_T>
    __device__ void computeStateDerivAllRollouts(DYN_T* dynamics, float* x_thread, float* u_thread, float* xdot_thread) {
        // The prior loop already guarantees that the global index is less than the number of rollouts
        dynamics->xDot(x_thread, u_thread, xdot_thread);
    }

    __device__ void incrementStateAllRollouts(int state_dim, int blocksize_y, int thread_idy, float dt,
                                                float* x_thread, float* xdot_thread) {
        // The prior loop already guarantees that the global index is less than the number of rollouts
        //Implementing simple first order Euler for now, more complex scheme can be added later
        for (int i = thread_idy; i < state_dim; i += blocksize_y) {
            x_thread[i] += xdot_thread[i] * dt;
            xdot_thread[i] = 0; // Reset the derivative to zero
        }
    }

    template<class COST_T>
    __device__ void computeAndSaveCost(int num_rollouts, int global_idx, COST_T* costs, float* x_thread,
                                        float running_cost, float* cost_rollouts_device) {
        if (global_idx < num_rollouts) {
            cost_rollouts_device[global_idx] = running_cost + costs->computeTerminalCost(x_thread);
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

    __device__ void strideControlWeightReduction(int num_rollouts, int num_timesteps, int sum_stride, int thread_idx,
            int block_idx, int control_dim, float* exp_costs_d, float normalizer, float* du_d, float* u, float* u_intermediate) {
        for (int i = 0; i < sum_stride; ++i) { // Iterate through the size of the subsection
            if ((thread_idx * sum_stride +i) < num_rollouts) { //Ensure we do not go out of bounds
                float weight = exp_costs_d[sum_stride*thread_idx + i] / normalizer; // compute the importance sampling weight
                for (int j = 0; j < control_dim; ++j) { // Iterate through the control dimensions
                    // Rollout index: (thread_idx*sum_stride + i)*(num_timesteps*control_dim)
                    // Current timestep: block_idx*control_dim
                    u[j] = du_d[(thread_idx*sum_stride + i)*(num_timesteps*control_dim) + block_idx*control_dim + j];
                    u_intermediate[thread_idx * control_dim + j] += weight * u[j];
                }
            }
        }
    }

    __device__ void rolloutWeightReductionAndSaveControl(int thread_idx, int block_idx, int num_rollouts, int num_timesteps, int control_dim, int sum_stride, float* u, float* u_intermediate, float* du_new_d) {
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
    template<class DYN_T, class COST_T, int NUM_ROLLOUTS, int BLOCKSIZE_X, int BLOCKSIZE_Y>
    void launchRolloutKernel(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, float* x_d, float* u_d, float* du_d, float* sigma_u_d, float* trajectory_costs, cudaStream_t stream) {
        const int gridsize_x = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;
        dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
        dim3 dimGrid(gridsize_x, 1, 1);
        rolloutKernel<DYN_T, COST_T><<<dimGrid, dimBlock, 0, stream>>>(dynamics, costs, dt,
                num_timesteps, x_d, u_d, du_d, sigma_u_d, trajectory_costs);
        CudaCheckError();
        HANDLE_ERROR( cudaStreamSynchronize(stream) );
    }

    void launchNormExpKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float gamma, float baseline) {
        dim3 dimBlock(blocksize_x, 1, 1);
        dim3 dimGrid((num_rollouts - 1) / blocksize_x + 1, 1, 1);
        normExpKernel<<<dimGrid, dimBlock>>>(num_rollouts, trajectory_costs_d, gamma, baseline);
        CudaCheckError();
        HANDLE_ERROR( cudaDeviceSynchronize() );
    }

    template<class DYN_T, int NUM_ROLLOUTS, int SUM_STRIDE >
    void launchWeightedReductionKernel(float* exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d, float normalizer, int num_timesteps) {
        dim3 dimBlock((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1, 1, 1);
        dim3 dimGrid(num_timesteps, 1, 1);
        weightedReductionKernel<DYN_T::CONTROL_DIM, NUM_ROLLOUTS, SUM_STRIDE><<<dimGrid, dimBlock>>>(exp_costs_d, du_d, sigma_u_d, du_new_d, normalizer, num_timesteps);
        CudaCheckError();
        HANDLE_ERROR( cudaDeviceSynchronize() );
    }



}
