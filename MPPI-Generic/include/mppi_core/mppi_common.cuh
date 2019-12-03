//
// Created by Manan Gandhi on 12/2/19.
//

#ifndef MPPIGENERIC_MPPI_COMMON_CUH
#define MPPIGENERIC_MPPI_COMMON_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

namespace mppi_common {

    // Kernel functions
    template<class DYN_T, class COST_T>
    __global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs);

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
    __device__ inline void loadGlobalToShared(int state_dim,
                                              int control_dim,
                                              float* x0_device,
                                              float* sigma_u_device,
                                              float* x_thread,
                                              float* xdot_thread,
                                              float* u_thread,
                                              float* du_thread,
                                              float* sigma_u_thread);

    __device__ inline void injectControlNoise(int i,
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
    __device__ inline void computeRunningCostAllRollouts(int thread_id, int num_rollouts, COST_T* costs,
                                                         float* x_thread, float* u_thread, float& running_cost) {
//        if (thread_id < num_rollouts) {
//            running_cost += costs->computeRunningCost_device(x_thread, u_thread);
//        }
    }

    template<class DYN_T>
    __device__ inline void computeStateDerivAllRollouts(int thread_id, int num_rollouts, DYN_T* dynamics,
                                                        float* x_thread, float* u_thread, float* xdot_thread) {
//        if (thread_id < num_rollouts) {
//            dynamics->computeStateDeriv(x_thread, u_thread, xdot_thread);
//        }
    }

    template<class DYN_T>
    __device__ inline void incrementStateAllRollouts(int thread_id, int num_rollouts, DYN_T* dynamics,
                                                     float* x_thread, float* xdot_thread) {
//        if (thread_id < num_rollouts) {
//            dynamics->incrementState(x_thread, xdot_thread);
//        }
    }

    template<class COST_T>
    __device__ inline void computeTerminalCostandSave(int thread_id, int num_rollouts, COST_T* costs,
                                                      float* x_thread, float running_cost, float* cost_rollouts_device) {
//        if (thread_id < num_rollouts) {
//            cost_rollouts_device[thread_id] = running_cost + costs->computeTermianalCost(x_thread);
//        }
    }

    template<class DYN_T, class COST_T>
    __global__ void rolloutKernel(DYN_T dynamics, COST_T costs) {
        int tdx = threadIdx.x;
        int tid = tdx + blockDim.x * blockIdx.x;
    }

}


#endif //MPPIGENERIC_MPPI_COMMON_CUH