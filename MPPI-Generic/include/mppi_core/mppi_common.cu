#include "mppi_core/mppi_common.cuh"

//#define state_dim DYN_T::STATE_DIM;

namespace mppi_common {
    // Kernel functions
    template<class DYN_T, class COST_T>
    __global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs, float dt,
                                    int num_timesteps, float* x_d, float* u_d, float* du_d, float* sigma_u_d) {
        //Get thread and block id
        int thread_idx = threadIdx.x;
        int thread_idy = threadIdx.y;
        int block_idx = blockIdx.x;
        int global_idx = blocksize_x*block_idx + thread_idx;

        //Create shared state and control arrays
        __shared__ float x_shared[blocksize_x*state_dim];
        __shared__ float xdot_shared[blocksize_x*state_dim];
        __shared__ float u_shared[blocksize_x*control_dim];
        __shared__ float du_shared[blocksize_x*control_dim];
        __shared__ float sigma_u_shared[blocksize_x*control_dim];

        //Create local state, state dot and controls
        float* x;
        float* xdot;
        float* u;
        float* du;
        float* sigma_u;

        //Initialize running cost and total cost
        float running_cost = 0;
        float* cost[num_rollouts];

        //Load global array to shared array
        if (global_idx < num_rollouts) {
            x = &x_shared[thread_idx * state_dim];
            xdot = &xdot_shared[thread_idx * state_dim];
            u = &u_shared[thread_idx * control_dim];
            du = &du_shared[thread_idx * control_dim];
            sigma_u = &sigma_u_shared[thread_idx * control_dim];
        }
        __syncthreads();
        loadGlobalToShared(global_idx, thread_idy, x_d, sigma_u_d,
                            x, xdot, u, du, sigma_u);
        __syncthreads();

        /*<----Start of simulation loop-----> */
        for (int t = 0; t < num_timesteps; t++) {
            if (global_idx < num_rollouts) {
                //Load noise trajectories scaled by the exploration factor
                injectControlNoise(t, global_idx, thread_idy, num_timesteps, num_rollouts, u_d, du_d, u, du, sigma_u);
                __syncthreads();

                //Accumulate running cost
                computeRunningCostAllRollouts(global_idx, costs, x, u, running_cost);
                __syncthreads();

                //Compute state derivatives
                computeStateDerivAllRollouts(global_idx, dynamics, x, u, xdot);
                __syncthreads();

                //Increment states
                incrementStateAllRollouts(global_idx, thread_idy, dynamics, x, xdot);
                __syncthreads();
            }
        }

        //Compute terminal cost and the final cost for each thread
        computeAndSaveCost(global_idx, costs, x, running_cost, cost);
        __syncthreads();
    }

    // Launch functions

    // RolloutKernel Helpers -------------------------------------------------------------------------------------------
    /*
     * loadGlobalToShared
     * Copy global memory into shared memory
     *
     * Args:
     * state_dim
     * control_dim
     * x0_device: initial condition in device memory
     * sigma_u_device: control exploration variance in device memory
     * x_thread: state in shared memory
     * xdot_thread: state_dot in shared memory
     * u_thread: control / perturbed control in shared memory
     * du_thread: control perturbation in shared memory
     * sigma_u_thread: control exploration variance in shared memory
     *
     */
    __device__ void loadGlobalToShared(int global_idx, int thread_idy,
                                        float* x_device, float* sigma_u_device, float* x_thread,
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

    __device__ void injectControlNoise(int t, int global_idx, int thread_idy, int num_timesteps,
                                        float* u_traj_device, float* ep_v_device, float* u_thread, float* du_thread,
                                        float* sigma_u_thread) {
        //Load the noise trajectory scaled by the exploration factor
        if (global_idx < num_rollouts) {
            for (int i = thread_idy; i < control_dim; i += blocksize_y) {
                //Keep one noise free trajectory
                if (global_idx == 0){
                    du_thread[i] = 0;
                    u_thread[i] = u_traj_device[t*control_dim + i];
                }
                //Generate 1% zero control trajectory
                else if (global_idx >= 0.99*num_rollouts) {
                    du_thread[i] = ep_v_device[global_idx*control_dim*num_timesteps + t*control_dim + i] * sigma_u_thread[i];
                    u_thread[i] = du_thread[i];
                }
                else {
                    du_thread[i] = ep_v_device[global_idx*control_dim*num_timesteps + t*control_dim + i] * sigma_u_thread[i];
                    u_thread[i] = u_traj_device[t*control_dim + i] + du_thread[i];
                }
            }
        }
    }

    template<class COST_T>
    __device__ void computeRunningCostAllRollouts(int global_idx, COST_T* costs,
                                                         float* x_thread, float* u_thread, float& running_cost) {
        if (global_idx < num_rollouts) {
            running_cost += costs->computeRunningCost(x_thread, u_thread);
        }
    }

    template<class DYN_T>
    __device__ void computeStateDerivAllRollouts(int global_idx, DYN_T* dynamics,
                                                        float* x_thread, float* u_thread, float* xdot_thread) {
        if (global_idx < num_rollouts) {
            dynamics->xDot(x_thread, u_thread, xdot_thread);
        }
    }

    template<class DYN_T>
    __device__ void incrementStateAllRollouts(int global_idx, int thread_idy, float dt,
                                                float* x_thread, float* xdot_thread) {
        if (global_idx < num_rollouts) {
            //Implementing simple first order Euler for now, more complex scheme can be added later
            for (int i = thread_idy; i < state_dim; i += blocksize_y) {
                x_thread[i] += xdot_thread[i] * dt;
            }
        }
    }

    template<class COST_T>
    __device__ void computeAndSaveCost(int global_idx, COST_T* costs, float* x_thread,
                                        float running_cost, float* cost_rollouts_device) {
        if (global_idx < num_rollouts) {
            cost_rollouts_device[global_idx] = running_cost + costs->computeTermianalCost(x_thread);
        }
    }

    // End of rollout kernel helpers -----------------------------------------------------------------------------------

    __global__ void normExpKernel(float* trajectory_costs_d, float gamma, float baseline) {
        int thread_idx = threadIdx.x;
        int block_idx = blockIdx.x;
        int global_idx = blocksize_x*block_idx + thread_idx;

        if (global_idx < num_rollouts) {
            float cost_dif = trajectory_costs_d[global_idx] - baseline;
            trajectory_costs_d[global_idx] = exp(-gamma*cost_dif);
        }
    }

    __global__ void weightedReductionKernel(float* exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d, float normalizer, int num_timesteps) {
        int thread_idx = threadIdx.x;
        int block_idx = blockIdx.x;

        //Create a shared array for intermediate sums
        __shared__ float u_intermediate[control_dim*((num_rollouts-1)/sum_stride) + 1];
        int stride = sum_stride;

        float u[control_dim];
        for (int i = 0; i < control_dim; i++) {
            u[i] = 0;
            u_intermediate[thread_idx*control_dim + i] = 0;
        }

        __syncthreads();

        //Sum the weighted control variations at a desired stride
        if (thread_idx*stride < num_rollouts) {
            float weight = 0;
            for (int i = 0; i < stride; i++) {
                weight = exp_costs_d[thread_idx*stride + i]/normalizer;
                for (int j = 0; j < control_dim; j++) {
                    u[j] = du_d[(thread_idx*stride + i)*(num_timesteps*control_dim) + block_idx*control_dim + j]*sigma_u_d[j];
                    u_intermediate[thread_idx*control_dim + j] += weight*u[j];
                }
            }
        }

        __syncthreads();

        //Sum all weighted control variations
        if (thread_idx == 0 && block_idx < num_timesteps) {
            for (int i = 0; i < control_dim; i++) {
                u[i] = 0;
            }
            for (int i = 0; i < ((num_rollouts - 1)/sum_stride + 1); i++) {
                for (int j = 0; j < control_dim; j++) {
                    u[j] += u_intermediate[i*control_dim + j];
                }
            }
            for (int i = 0; i < control_dim; i++) {
                du_new_d[block_idx*control_dim + i] = u[i];
            }
        }
    }

    void launchWeightedReductionKernel(float* exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d, float normalizer, int num_timesteps) {
        dim3 dimBlock((num_rollouts-1)/sum_stride + 1, 1, 1);
        dim3 dimGrid(num_timesteps, 1, 1);
        weightedReductionKernel<<<dimGrid, dimBlock>>>(exp_costs_d, du_d, sigma_u_d, du_new_d, normalizer, num_timesteps);
        CudaCheckError();
        HANDLE_ERROR( cudaDeviceSynchronize() );
    }

    void launchNormExpKernel(float* trajectory_costs_d, float gamma, float baseline) {
        dim3 dimBlock(blocksize_x, 1, 1);
        dim3 dimGrid((num_rollouts-1)/blocksize_x + 1, 1, 1);
        normExpKernel<<<dimGrid, dimBlock>>>(trajectory_costs_d, gamma, baseline);
        CudaCheckError();
        HANDLE_ERROR( cudaDeviceSynchronize() );
    }

    template<class DYN_T, class COST_T>
    void launchRolloutKernel(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, float* x_d, float* u_d, float* du_d, float* sigma_u_d) {
        const int gridsize_x = (num_rollouts - 1)/blocksize_x + 1;
        dim3 dimBlock(blocksize_x, blocksize_y, 1);
        dim3 dimGrid(gridsize_x, 1, 1);
        rolloutKernel<DYN_T, COST_T><<<dimGrid, dimBlock>>>(dynamics, costs, dt,
                num_timesteps, x_d, u_d, du_d, sigma_u_d);
        CudaCheckError();
        HANDLE_ERROR( cudaDeviceSynchronize() );
    }

}
