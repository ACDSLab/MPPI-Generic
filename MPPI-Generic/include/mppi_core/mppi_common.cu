#include "mppi_core/mppi_common.cuh"

#define state_dim 12
#define control_dim 3
#define blocksize_x 64
#define blocksize_y 8
#define num_rollouts 2000

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

    // RolloutKernel Helpers
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
        if (global_idx < num_rollouts) {
            for (int i = thread_idy; i < state_dim; i += blocksize_y) {
                x_thread[i] = x_device[i];
                xdot_thread[i] = 0;
            }
            for (int i = thread_idy; i < control_dim; i += blocksize_y) {
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


//    void launchRolloutKernel() {
//
//    }

}