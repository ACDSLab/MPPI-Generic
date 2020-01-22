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
     * global_idx
     * thread_dy
     * x0_device: initial condition in device memory
     * sigma_u_device: control exploration variance in device memory
     * x_thread: state in shared memory
     * xdot_thread: state_dot in shared memory
     * u_thread: control / perturbed control in shared memory
     * du_thread: control perturbation in shared memory
     * sigma_u_thread: control exploration variance in shared memory
     *
     */
    __device__ void loadGlobalToShared(int global_idx,
                                       int thread_idy,
                                       float* x0_device,
                                       float* sigma_u_device,
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
     * CONTROL_DIM: Number of controls, defined in DYN_T
     * BLOCKSIZE_Y: Y dimension of each block of threads
     * NUM_ROLLOUTS: Total number of rollouts
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

    template<class COST_T>
    __device__ void computeRunningCostAllRollouts(int thread_id, int num_rollouts, COST_T* costs,
                                                         float* x_thread, float* u_thread, float& running_cost);

    template<class DYN_T>
    __device__ void computeStateDerivAllRollouts(int thread_id, DYN_T* dynamics,
                                                        float* x_thread, float* u_thread, float* xdot_thread);

    template<class DYN_T>
    __device__ void incrementStateAllRollouts(int thread_id, int num_rollouts, float dt,
                                                     float* x_thread, float* xdot_thread);

    template<class COST_T>
    __device__ void computeAndSaveCost(int thread_id, int num_rollouts, COST_T* costs,
                                                      float* x_thread, float running_cost, float* cost_rollouts_device);

    __global__ void normExpKernel(float* trajectory_costs_d, float gamma, float baseline);

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
