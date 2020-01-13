//
// Created by Manan Gandhi on 12/2/19.
//

#ifndef MPPIGENERIC_MPPI_COMMON_CUH
#define MPPIGENERIC_MPPI_COMMON_CUH

#include <curand.h>
#include "utils/gpu_err_chk.cuh"

namespace mppi_common {

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
    __device__ void loadGlobalToShared(int state_dim,
                                              int control_dim,
                                              float* x0_device,
                                              float* sigma_u_device,
                                              float* x_thread,
                                              float* xdot_thread,
                                              float* u_thread,
                                              float* du_thread,
                                              float* sigma_u_thread);

    __device__ void injectControlNoise(int i,
                                              int control_dim,
                                              int thread_id,
                                              int num_timesteaps,
                                              int num_rollouts,
                                              float* u_traj_device,
                                              float* ep_v_device,
                                              float* u_thread,
                                              float* du_thread,
                                              float* sigma_u_thread);

    template<class COST_T>
    __device__ void computeRunningCostAllRollouts(int thread_id, int num_rollouts, COST_T* costs,
                                                         float* x_thread, float* u_thread, float& running_cost);

    template<class DYN_T>
    __device__ void computeStateDerivAllRollouts(int thread_id, int num_rollouts, DYN_T* dynamics,
                                                        float* x_thread, float* u_thread, float* xdot_thread);

    template<class DYN_T>
    __device__ void incrementStateAllRollouts(int thread_id, int num_rollouts, DYN_T* dynamics,
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