//
// Created by Manan Gandhi on 12/2/19.
//

#ifndef MPPIGENERIC_MPPI_COMMON_CUH
#define MPPIGENERIC_MPPI_COMMON_CUH

#include <curand.h>
#include "utils/gpu_err_chk.cuh"

namespace mppi_common {

    const int STATE_DIM = 12;
    const int CONTROL_DIM = 3;

    const int BLOCKSIZE_X = 64;
    const int BLOCKSIZE_Y = 8;
    const int NUM_ROLLOUTS = 2000;
    const int SUM_STRIDE = 128;

    // Kernel functions
    template<class DYN_T, class COST_T>
    __global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs, float dt,
                                  int num_timesteps, float* x_d, float* u_d, float* du_d, float* sigma_u_d);

    // Launch functions

    // RolloutKernel Helpers
    /*
     * loadGlobalToShared
     * Copy global memory into shared memory
     *
     * Args:
     * state_dim: Number of states, defined in DYN_T
     * control_dim: Number of controls, defined in DYN_T
     * num_rollouts: Total number of rollouts
     * blocksize_y: Y dimension of each block of threads
     * global_idx: Current rollout index.
     * thread_idy: Current y index of block dimension.
     * x0_device: initial condition in device memory
     * sigma_u_device: control exploration variance in device memory
     * x_thread: state in shared memory
     * xdot_thread: state_dot in shared memory
     * u_thread: control / perturbed control in shared memory
     * du_thread: control perturbation in shared memory
     * sigma_u_thread: control exploration variance in shared memory
     *
     */
    __device__ void loadGlobalToShared(int state_dim,
                                       int control_dim,
                                       int num_rollouts,
                                       int blocksize_y,
                                       int global_idx,
                                       int thread_idy,
                                       const float* x0_device,
                                       const float* sigma_u_device,
                                       float* x_thread,
                                       float* xdot_thread,
                                       float* u_thread,
                                       float* du_thread,
                                       float* sigma_u_thread);
    /*
     * injectControlNoise
     * Disturb control trajectories per timestep
     *
     * Args:
     * control_dim: Number of controls, defined in DYN_T
     * blocksize_y: Y dimension of each block of threads
     * num_rollouts: Total number of rollouts
     * num_timesteps: Trajectory length
     * current_timestep: Index of time in current trajectory
     * global_idx: Current rollout index.
     * thread_idy: Current y index of block dimension.
     * u_traj_device: Complete control trajectory
     * ep_v_device: Complete set of disturbances for all rollouts and timesteps
     * u_thread: Current control for the given rollout
     * du_thread: Current disturbance for the given rollout
     * sigma_u_thread: Control variance for all rollouts
     */
    __device__ void injectControlNoise(int control_dim,
                                       int blocksize_y,
                                       int num_rollouts,
                                       int num_timesteps,
                                       int current_timestep,
                                       int global_idx,
                                       int thread_idy,
                                       const float* u_traj_device,
                                       const float* ep_v_device,
                                       const float* sigma_u_thread,
                                       float* u_thread,
                                       float* du_thread);

    /*
     * computeRunningCostAllRollouts
     * Compute the running cost for each rollout
     *
     * Args:
     * costs: cost function class
     * x_thread: Current state for the given rollout
     * u_thread: Current control for the given rollout,
     * running_cost: Running cost for the given rollout
     */
    template<class COST_T>
    __device__ void computeRunningCostAllRollouts(COST_T* costs, float* x_thread, float* u_thread, float& running_cost);

    /*
     * computeRunningCostAllRollouts
     * Compute the running cost for each rollout
     *
     * Args:
     * dynamics: dynamics function class
     * x_thread: Current state for the given rollout
     * u_thread: Current control for the given rollout,
     * xdot_thread: State derivative for the given rollout
     */
    template<class DYN_T>
    __device__ void computeStateDerivAllRollouts(DYN_T* dynamics, float* x_thread, float* u_thread, float* xdot_thread);

    /*
     * incrementStateAllRollouts
     * Increment the state using the forward Euler method
     *
     * Args:
     * state_dim: Number of states, defined in DYN_T
     * blocksize_y: Y dimension of each block of threads
     * thread_idy: Current y index of block dimension.
     * dt: Timestep size
     * x_thread: Current state for the given rollout
     * xdot_thread: State derivative for the given rollout
     */
    template<class DYN_T>
    __device__ void incrementStateAllRollouts(int state_dim, int blocksize_y, int thread_idy, float dt,
                                                     float* x_thread, float* xdot_thread);

    template<class COST_T>
    __device__ void computeAndSaveCost(int num_rollouts, int global_idx, COST_T* costs,
                                                      float* x_thread, float running_cost, float* cost_rollouts_device);

    __global__ void normExpKernel(float* trajectory_costs_d, float gamma, float baseline);

    template<class DYN_T, int num_rollouts, int sum_stride>
    __global__ void weightedReductionKernel(float* exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d, float normalizer, int num_timesteps);

    template<class DYN_T, class COST_T>
    void launchRolloutKernel(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, float* x_d, float* u_d, float* du_d, float* sigma_u_d);

    void launchNormExpKernel(float* trajectory_costs_d, float gamma, float baseline);

    void launchWeightedReductionKernel(float* exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d, float normalizer, int num_timesteps);

}
#if __CUDACC__
#include "mppi_common.cu"
#endif

#endif //MPPIGENERIC_MPPI_COMMON_CUH
