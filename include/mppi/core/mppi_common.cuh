//
// Created by Manan Gandhi on 12/2/19.
//

#ifndef MPPIGENERIC_MPPI_COMMON_CUH
#define MPPIGENERIC_MPPI_COMMON_CUH

namespace mppi_common {


    // Kernel functions
    template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y,
             int NUM_ROLLOUTS, int BLOCKSIZE_Z = 1>
    __global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs, float dt,
                                  int num_timesteps, float* x_d, float* u_d, float* du_d, float* sigma_u_d,
                                  float* trajectory_costs_d);

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
                                       int thread_idz,
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
                                       float* ep_v_device,
                                       const float* sigma_u_thread,
                                       float* u_thread,
                                       float* du_thread);

    template<class COST_T>
    __device__ void computeAndSaveCost(int num_rollouts, int global_idx, COST_T* costs, float* x_thread,
                                       float running_cost, float* cost_rollouts_device);

    // Norm Exponential Kernel

    __global__ void normExpKernel(int num_rollouts, float* trajectory_costs_d, float gamma, float baseline);

    // Norm Exp Kernel Helpers
    float computeBaselineCost(float* cost_rollouts_host, int num_rollouts);

    float computeNormalizer(float* cost_rollouts_host, int num_rollouts);

    /**
     * Calculates the free energy mean and variance from the different
     * cost trajectories after normExpKernel
     * Inputs:
     *  cost_rollouts_host - sampled cost trajectories
     *  num_rollouts - the number of sampled cost trajectories
     *  lambda - the lambda term from the definition of free energy
     *  baseline - minimum cost trajectory
     * Outputs:
     *  free_energy - the free energy of the samples
     *  free_energy_var - the variance of the free energy calculation
     */
    void computeFreeEnergy(float& free_energy, float& free_energy_var,
                           float* cost_rollouts_host,  int num_rollouts,
                           float baseline, float lambda = 1.0);

    // Weighted Reduction Kernel
    template<int CONTROL_DIM, int NUM_ROLLOUT, int SUM_STRIDE>
    __global__ void weightedReductionKernel(float* exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d, float normalizer, int num_timesteps);

    // Weighted Reduction Kernel Helpers
    __device__ void setInitialControlToZero(int control_dim, int thread_idx, float* u, float* u_intermediate);

    __device__ void strideControlWeightReduction(int num_rollouts, int num_timesteps, int sum_stride, int thread_idx,
            int block_idx, int control_dim, float* exp_costs_d, float normalizer, float* du_d, float* u, float* u_intermediate);

    __device__ void rolloutWeightReductionAndSaveControl(int thread_idx, int block_idx, int num_rollouts, int num_timesteps,
            int control_dim, int sum_stride, float* u, float* u_intermediate, float* du_new_d);

    // Launch functions
    template<class DYN_T, class COST_T>
    void launchRolloutKernel(DYN_T* dynamics, COST_T* costs, float dt, int num_timesteps, float* x_d, float* u_d,
            float* du_d, float* sigma_u_d, float* trajectory_costs, cudaStream_t stream);

    void launchNormExpKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float gamma, float baseline, cudaStream_t stream);

    template<class DYN_T, int NUM_ROLLOUTS, int SUM_STRIDE >
    void launchWeightedReductionKernel(float* exp_costs_d, float* du_d, float* sigma_u_d, float* du_new_d, float normalizer, int num_timesteps, cudaStream_t stream);

}

namespace rmppi_kernels {
  template <class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int SAMPLES_PER_CONDITION>
  __global__ void initEvalKernel(DYN_T* dynamics,
                                 COST_T* costs,
                                 int num_timesteps,
                                 int ctrl_stride,
                                 float dt,
                                 int* strides_d,
                                 float* exploration_var_d,
                                 float* states_d,
                                 float* control_d,
                                 float* control_noise_d,
                                 float* costs_d);

  template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int SAMPLES_PER_CONDITION>
  void launchInitEvalKernel(DYN_T* dynamics,
                            COST_T* costs,
                            int num_candidates,
                            int num_timesteps,
                            int ctrl_stride,
                            float dt,
                            int* strides_d,
                            float* exploration_var_d,
                            float* states_d,
                            float* control_d,
                            float* control_noise_d,
                            float* costs_d);

  template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y,
      int NUM_ROLLOUTS, int BLOCKSIZE_Z = 2>
  __global__ void RMPPIRolloutKernel(DYN_T * dynamics, COST_T* costs,
                                     float dt,
                                     int num_timesteps,
                                     float lambda,
                                     float value_func_threshold,
                                     float* x_d,
                                     float* u_d,
                                     float* du_d,
                                     float* feedback_gains_d,
                                     float* sigma_u_d,
                                     float* trajectory_costs_d);

  template<class DYN_T, class COST_T, int NUM_ROLLOUTS, int BLOCKSIZE_X,
           int BLOCKSIZE_Y, int BLOCKSIZE_Z = 2>
  void launchRMPPIRolloutKernel(DYN_T* dynamics, COST_T* costs,
                                float dt,
                                int num_timesteps,
                                float lambda,
                                float value_func_threshold,
                                float* x_d,
                                float* u_d,
                                float* du_d,
                                float* feedback_gains_d,
                                float* sigma_u_d,
                                float* trajectory_costs,
                                cudaStream_t stream);
}
#if __CUDACC__
#include "mppi_common.cu"
#endif

#endif //MPPIGENERIC_MPPI_COMMON_CUH
