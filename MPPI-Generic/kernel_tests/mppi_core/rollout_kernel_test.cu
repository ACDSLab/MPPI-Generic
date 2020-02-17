#include "rollout_kernel_test.cuh"

#include <dynamics/cartpole/cartpole_dynamics.cuh>
#include <cost_functions/cartpole/cartpole_quadratic_cost.cuh>
const int STATE_DIM = 12;
const int CONTROL_DIM = 3;
const int NUM_ROLLOUTS = 100; // .99 times this number has to be an integer... TODO fix how brittle this is
const int BLOCKSIZE_X = 64;
const int BLOCKSIZE_Y = 8; // Blocksize_y has to be greater than the control dim TODO fix how we step through the controls

__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device,
        float* x_thread_device, float* xdot_thread_device, float* u_thread_device, float* du_thread_device, float* sigma_u_thread_device) {
    int thread_idx = threadIdx.x;
    int thread_idy = threadIdx.y;
    int block_idx = blockIdx.x;
    int global_idx = threadIdx.x + block_idx*blockDim.x;

    //Create shared arrays which hold state and control data
    __shared__ float x_shared[BLOCKSIZE_X * STATE_DIM];
    __shared__ float xdot_shared[BLOCKSIZE_X * STATE_DIM];
    __shared__ float u_shared[BLOCKSIZE_X * CONTROL_DIM];
    __shared__ float du_shared[BLOCKSIZE_X * CONTROL_DIM];
    __shared__ float sigma_u_shared[BLOCKSIZE_X * CONTROL_DIM];

    float* x_thread;
    float* xdot_thread;

    float* u_thread;
    float* du_thread;
    float* sigma_u_thread;

    if (global_idx < NUM_ROLLOUTS) {
        x_thread = &x_shared[thread_idx * STATE_DIM];
        xdot_thread = &xdot_shared[thread_idx * STATE_DIM];
        u_thread = &u_shared[thread_idx * CONTROL_DIM];
        du_thread = &du_shared[thread_idx * CONTROL_DIM];
        sigma_u_thread = &sigma_u_shared[thread_idx * CONTROL_DIM];
    }
    __syncthreads();
    mppi_common::loadGlobalToShared(STATE_DIM, CONTROL_DIM, NUM_ROLLOUTS,
            BLOCKSIZE_Y, global_idx, thread_idy,
            x0_device, sigma_u_device, x_thread, xdot_thread, u_thread, du_thread, sigma_u_thread);
    __syncthreads();

    // Check if on the first rollout the correct values were coped over
    if (global_idx == 1) {
        for (int i = 0; i < STATE_DIM; ++i) {
            x_thread_device[i] = x_thread[i];
            xdot_thread_device[i] = xdot_thread[i];
        }

        for (int i = 0; i < CONTROL_DIM; ++i) {
            u_thread_device[i] = u_thread[i];
            du_thread_device[i] = du_thread[i];
            sigma_u_thread_device[i] = sigma_u_thread[i];
        }
    }



    // To test what the results are, we have to return them back to the host.
}

void launchGlobalToShared_KernelTest(const std::vector<float>& x0_host,const std::vector<float>& u_var_host,
        std::vector<float>& x_thread_host, std::vector<float>& xdot_thread_host,
        std::vector<float>& u_thread_host, std::vector<float>& du_thread_host, std::vector<float>& sigma_u_thread_host ) {

    // Define the initial condition x0_device and the exploration variance in global device memory
    float* x0_device;
    float* u_var_device;
    HANDLE_ERROR(cudaMalloc((void**)&x0_device, sizeof(float)*STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&u_var_device, sizeof(float)*CONTROL_DIM));

    HANDLE_ERROR(cudaMemcpy(x0_device, x0_host.data(), sizeof(float)*STATE_DIM, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(u_var_device, u_var_host.data(), sizeof(float)*CONTROL_DIM, cudaMemcpyHostToDevice));


    // Define the return arguments in global device memory
    float* x_thread_device;
    float* xdot_thread_device;
    float* u_thread_device;
    float* du_thread_device;
    float* sigma_u_thread_device;

    HANDLE_ERROR(cudaMalloc((void**)&x_thread_device, sizeof(float)*STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&xdot_thread_device, sizeof(float)*STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&u_thread_device, sizeof(float)*CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&du_thread_device, sizeof(float)*CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&sigma_u_thread_device, sizeof(float)*CONTROL_DIM));

    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 dimGrid(2048);

    loadGlobalToShared_KernelTest<<<dimGrid,dimBlock>>>(x0_device, u_var_device,
            x_thread_device, xdot_thread_device, u_thread_device, du_thread_device, sigma_u_thread_device);
    CudaCheckError();

    // Copy the data back to the host
    HANDLE_ERROR(cudaMemcpy(x_thread_host.data(), x_thread_device, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(xdot_thread_host.data(), xdot_thread_device, sizeof(float)*STATE_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u_thread_host.data(), u_thread_device, sizeof(float)*CONTROL_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(du_thread_host.data(), du_thread_device, sizeof(float)*CONTROL_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(sigma_u_thread_host.data(), sigma_u_thread_device, sizeof(float)*CONTROL_DIM, cudaMemcpyDeviceToHost));

    // Free the cuda memory that we allocated
    cudaFree(x0_device);
    cudaFree(u_var_device);

    cudaFree(x_thread_device);
    cudaFree(xdot_thread_device);
    cudaFree(u_thread_device);
    cudaFree(du_thread_device);
    cudaFree(sigma_u_thread_device);
}

__global__ void  injectControlNoiseOnce_KernelTest(int num_rollouts, int num_timesteps, int timestep, float* u_traj_device, float* ep_v_device, float* sigma_u_device, float* control_compute_device) {
    int global_idx = threadIdx.x + blockDim.x*blockIdx.x;
    int thread_idy = threadIdx.y;
    float u_thread[CONTROL_DIM];
    float du_thread[CONTROL_DIM];

    if (global_idx < num_rollouts) {
        mppi_common::injectControlNoise(CONTROL_DIM, BLOCKSIZE_Y, num_rollouts, num_timesteps, timestep, global_idx, thread_idy, u_traj_device, ep_v_device, sigma_u_device, u_thread, du_thread);
        if (thread_idy < CONTROL_DIM) {
            control_compute_device[global_idx * CONTROL_DIM + thread_idy] = u_thread[thread_idy];
        }
    }
}

void launchInjectControlNoiseOnce_KernelTest(const std::vector<float>& u_traj_host, const int num_rollouts, const int num_timesteps,
        std::vector<float>& ep_v_host, std::vector<float>& sigma_u_host, std::vector<float>& control_compute) {

    // Timestep
    int timestep = 0;

    // Declare variables for device memory
    float* u_traj_device;
    float* ep_v_device;
    float* sigma_u_device;
    float* control_compute_device;

    // Allocate cuda memory
    HANDLE_ERROR(cudaMalloc((void**)&u_traj_device, sizeof(float)*u_traj_host.size()));
    HANDLE_ERROR(cudaMalloc((void**)&ep_v_device, sizeof(float)*ep_v_host.size()));
    HANDLE_ERROR(cudaMalloc((void**)&sigma_u_device, sizeof(float)*sigma_u_host.size()));
    HANDLE_ERROR(cudaMalloc((void**)&control_compute_device, sizeof(float)*control_compute.size()));

    // Copy the control trajectory and the control variance to the device
    HANDLE_ERROR(cudaMemcpy(u_traj_device, u_traj_host.data(), sizeof(float)*u_traj_host.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(sigma_u_device, sigma_u_host.data(), sizeof(float)*sigma_u_host.size(), cudaMemcpyHostToDevice));

    // Generate the noise
    curandGenerator_t gen_;
    curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL);
    curandGenerateNormal(gen_, ep_v_device, ep_v_host.size(), 0.0, 1.0);

    // Copy the noise back to the host
    HANDLE_ERROR(cudaMemcpy(ep_v_host.data(), ep_v_device, sizeof(float)*ep_v_host.size(), cudaMemcpyDeviceToHost));

    // Create the block and grid dimensions
    dim3 block_size(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 grid_size(num_rollouts, 1);

    // Launch the test kernel
    injectControlNoiseOnce_KernelTest<<<grid_size,block_size>>>(num_rollouts, num_timesteps, timestep, u_traj_device, ep_v_device, sigma_u_device, control_compute_device);
    CudaCheckError();

    // Copy the result back to the host
    HANDLE_ERROR(cudaMemcpy(control_compute.data(), control_compute_device, sizeof(float)*control_compute.size(), cudaMemcpyDeviceToHost));

    // Free cuda memory
    cudaFree(u_traj_device);
    cudaFree(ep_v_device);
    cudaFree(control_compute_device);
    cudaFree(sigma_u_device);
    curandDestroyGenerator(gen_);
}

template<int control_dim, int blocksize_y>
__global__ void injectControlNoiseCheckControlV_KernelTest(int num_rollouts, int num_timesteps, int timestep,
        float* u_traj_device, float* ep_v_device, float* sigma_u_device) {
    int global_idx = threadIdx.x + blockDim.x*blockIdx.x;
    int thread_idy = threadIdx.y;
    float u_thread[control_dim];
    float du_thread[control_dim];

    if (global_idx < num_rollouts) {
        mppi_common::injectControlNoise(control_dim, blocksize_y, num_rollouts, num_timesteps, timestep, global_idx, thread_idy, u_traj_device, ep_v_device, sigma_u_device, u_thread, du_thread);
        __syncthreads();
    }
}

template<int num_rollouts, int num_timesteps, int control_dim, int blocksize_x, int blocksize_y, int gridsize_x>
void launchInjectControlNoiseCheckControlV_KernelTest(const std::array<float, num_timesteps*control_dim>& u_traj_host,
        std::array<float, num_rollouts*num_timesteps*control_dim>& ep_v_host, const std::array<float, control_dim>& sigma_u_host) {

    // Declare variables for device memory
    float* u_traj_device;
    float* ep_v_device;
    float* sigma_u_device;

    // Allocate cuda memory
    HANDLE_ERROR(cudaMalloc((void**)&u_traj_device, sizeof(float)*u_traj_host.size()));
    HANDLE_ERROR(cudaMalloc((void**)&ep_v_device, sizeof(float)*ep_v_host.size()));
    HANDLE_ERROR(cudaMalloc((void**)&sigma_u_device, sizeof(float)*sigma_u_host.size()));

    // Copy to device
    HANDLE_ERROR(cudaMemcpy(u_traj_device, u_traj_host.data(), sizeof(float)*u_traj_host.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(ep_v_device, ep_v_host.data(), sizeof(float)*ep_v_host.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(sigma_u_device, sigma_u_host.data(), sizeof(float)*sigma_u_host.size(), cudaMemcpyHostToDevice));


    dim3 dimBlock(blocksize_x, blocksize_y, 1);
    dim3 dimGrid(gridsize_x, 1, 1);

    for (int i = 0; i < num_timesteps; ++i) {
        injectControlNoiseCheckControlV_KernelTest<control_dim, blocksize_y><<<dimGrid, dimBlock>>>(num_rollouts, num_timesteps, i, u_traj_device, ep_v_device, sigma_u_device);
        CudaCheckError();
    }

    // Copy to the host
    HANDLE_ERROR(cudaMemcpy(ep_v_host.data(), ep_v_device, sizeof(float)*ep_v_host.size(), cudaMemcpyDeviceToHost));


    cudaFree(u_traj_device);
    cudaFree(ep_v_device);
    cudaFree(sigma_u_device);
}

template<class COST_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int STATE_DIM, int CONTROL_DIM>
__global__ void computeRunningCostAllRollouts_KernelTest(COST_T* cost_d, float dt, float* x_trajectory_d, float* u_trajectory_d, float* du_trajectory_d, float* var_d, float* cost_allrollouts_d) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x; // index on rollouts
    if (tid < NUM_ROLLOUTS) {
        float current_cost = 0.f;
        for (int t = 0; t < NUM_TIMESTEPS; ++t) {
            mppi_common::computeRunningCostAllRollouts(cost_d, dt, &x_trajectory_d[STATE_DIM*NUM_TIMESTEPS*tid + STATE_DIM*t],
                                          &u_trajectory_d[CONTROL_DIM*NUM_TIMESTEPS*tid + CONTROL_DIM*t],
                                          &du_trajectory_d[CONTROL_DIM*NUM_TIMESTEPS*tid + CONTROL_DIM*t],
                                          var_d, current_cost, t);
        }
        cost_allrollouts_d[tid] = current_cost;
    }
}

template<class COST_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int STATE_DIM, int CONTROL_DIM>
void computeRunningCostAllRollouts_CPU_TEST(COST_T& cost,
                                            float dt,
                                            std::array<float, STATE_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS>& x_trajectory,
                                            std::array<float, CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS>& u_trajectory,
                                            std::array<float, CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS>& du_trajectory,
                                            std::array<float, CONTROL_DIM>& sigma_u,
                                            std::array<float, NUM_ROLLOUTS>& cost_allrollouts) {
    float current_cost;
    for (int i = 0; i < NUM_ROLLOUTS; ++i) {
        current_cost = 0;
        for (int t = 0; t < NUM_TIMESTEPS; ++t) {
            current_cost += cost.computeRunningCost(&x_trajectory[STATE_DIM*NUM_TIMESTEPS*i + STATE_DIM*t],
                                                    &u_trajectory[CONTROL_DIM*NUM_TIMESTEPS*i + CONTROL_DIM*t],
                                                    &du_trajectory[CONTROL_DIM*NUM_TIMESTEPS*i + CONTROL_DIM*t],
                                                    sigma_u.data(), t)*dt;
        }
        cost_allrollouts[i] = current_cost;
    }
}

template<class COST_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int STATE_DIM, int CONTROL_DIM>
void launchComputeRunningCostAllRollouts_KernelTest(const COST_T& cost,
                                                    float dt,
                                                    const std::array<float, STATE_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS>& x_trajectory,
                                                    const std::array<float, CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS>& u_trajectory,
                                                    const std::array<float, CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS>& du_trajectory,
                                                    const std::array<float, CONTROL_DIM>& sigma_u,
                                                    std::array<float, NUM_ROLLOUTS>& cost_allrollouts) {
    // Declare variables for device memory
    float* x_traj_d;
    float* u_traj_d;
    float* du_traj_d;
    float* sigma_u_d;
    float* cost_allrollouts_d;


    // Allocate cuda memory
    HANDLE_ERROR(cudaMalloc((void**)&x_traj_d, sizeof(float)*x_trajectory.size()));
    HANDLE_ERROR(cudaMalloc((void**)&u_traj_d, sizeof(float)*u_trajectory.size()));
    HANDLE_ERROR(cudaMalloc((void**)&du_traj_d, sizeof(float)*du_trajectory.size()));
    HANDLE_ERROR(cudaMalloc((void**)&sigma_u_d, sizeof(float)*sigma_u.size()));
    HANDLE_ERROR(cudaMalloc((void**)&cost_allrollouts_d, sizeof(float)*cost_allrollouts.size()));


    // Copy the trajectories to the device
    HANDLE_ERROR(cudaMemcpy(x_traj_d, x_trajectory.data(), sizeof(float)*x_trajectory.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(u_traj_d, u_trajectory.data(), sizeof(float)*u_trajectory.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(du_traj_d, du_trajectory.data(), sizeof(float)*du_trajectory.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(sigma_u_d, sigma_u.data(), sizeof(float)*sigma_u.size(), cudaMemcpyHostToDevice));


    // Launch the test kernel
    computeRunningCostAllRollouts_KernelTest<COST_T, NUM_ROLLOUTS, NUM_TIMESTEPS, STATE_DIM, CONTROL_DIM><<<1,NUM_ROLLOUTS>>>(cost.cost_d_, dt, x_traj_d, u_traj_d, du_traj_d, sigma_u_d, cost_allrollouts_d);
    CudaCheckError();

    // Copy the result back to the host
    HANDLE_ERROR(cudaMemcpy(cost_allrollouts.data(), cost_allrollouts_d, sizeof(float)*cost_allrollouts.size(), cudaMemcpyDeviceToHost));
}

template<class DYN_T, int NUM_ROLLOUTS>
__global__ void computeStateDerivAllRollouts_KernelTest(DYN_T* dynamics_d, float* x_trajectory_d, float* u_trajectory_d, float* xdot_trajectory_d) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x; // index on rollouts
  //Create a shared array for the dynamics model to use
  // TODO this is needed since 0 size is a compile error
  // __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X + 1];
  // TODO Figure out how to incorporate BLOCKSIZE_X
  __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK + 1];
  if (tid < NUM_ROLLOUTS) {
            mppi_common::computeStateDerivAllRollouts(dynamics_d, &x_trajectory_d[DYN_T::STATE_DIM*tid],
                                                       &u_trajectory_d[DYN_T::CONTROL_DIM*tid],
                                                       &xdot_trajectory_d[DYN_T::STATE_DIM*tid],
                                                       theta_s);
    }
}

template<class DYN_T, int NUM_ROLLOUTS>
void launchComputeStateDerivAllRollouts_KernelTest(const DYN_T& dynamics,
                                                   const std::array<float, DYN_T::STATE_DIM*NUM_ROLLOUTS>& x_trajectory,
                                                   const std::array<float, DYN_T::CONTROL_DIM*NUM_ROLLOUTS>& u_trajectory,
                                                   std::array<float, DYN_T::STATE_DIM*NUM_ROLLOUTS>& xdot_trajectory) {
    // Declare variables for device memory
    float* x_traj_d;
    float* u_traj_d;
    float* xdot_traj_d;


    // Allocate cuda memory
    HANDLE_ERROR(cudaMalloc((void**)&x_traj_d, sizeof(float)*x_trajectory.size()));
    HANDLE_ERROR(cudaMalloc((void**)&u_traj_d, sizeof(float)*u_trajectory.size()));
    HANDLE_ERROR(cudaMalloc((void**)&xdot_traj_d, sizeof(float)*xdot_trajectory.size()));


    // Copy the trajectories to the device
    HANDLE_ERROR(cudaMemcpy(x_traj_d, x_trajectory.data(), sizeof(float)*x_trajectory.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(u_traj_d, u_trajectory.data(), sizeof(float)*u_trajectory.size(), cudaMemcpyHostToDevice));


    // Launch the test kernel
    computeStateDerivAllRollouts_KernelTest<DYN_T, NUM_ROLLOUTS><<<1,NUM_ROLLOUTS>>>(dynamics.model_d_, x_traj_d, u_traj_d, xdot_traj_d);
    CudaCheckError();

    // Copy the result back to the host
    HANDLE_ERROR(cudaMemcpy(xdot_trajectory.data(), xdot_traj_d, sizeof(float)*xdot_trajectory.size(), cudaMemcpyDeviceToHost));
}

__global__ void incrementStateAllRollouts_KernelTest(int state_dim, int num_rollouts, float dt, float* x_trajectory_d, float* xdot_trajectory_d) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x; // index on rollouts
    int blocksize_y = blockDim.y;
    int thread_idy = threadIdx.y;
//    printf("TID: %d\n", tid);
//    printf("blocksize_y: %d\n", blocksize_y);
//    printf("thread_idy: %d\n", thread_idy);
    if (tid < num_rollouts) {
//        if (tid == 1) {
//            printf("Current state [%f, %f, %f, %f]\n", x_trajectory_d[state_dim * tid],
//                   x_trajectory_d[state_dim * tid + 1], x_trajectory_d[state_dim * tid + 2],
//                   x_trajectory_d[state_dim * tid + 3]);
//            printf("Current xdot [%f, %f, %f, %f]\n", xdot_trajectory_d[state_dim * tid],
//                   xdot_trajectory_d[state_dim * tid + 1], xdot_trajectory_d[state_dim * tid + 2],
//                   xdot_trajectory_d[state_dim * tid + 3]);
//        }
        mppi_common::incrementStateAllRollouts(state_dim, blocksize_y, thread_idy, dt,
                &x_trajectory_d[state_dim * tid],&xdot_trajectory_d[state_dim * tid]);
//        if (tid == 1) {
//            printf("Post increment state [%f, %f, %f, %f]\n", x_trajectory_d[state_dim * tid],
//                   x_trajectory_d[state_dim * tid + 1], x_trajectory_d[state_dim * tid + 2],
//                   x_trajectory_d[state_dim * tid + 3]);
//            printf("Post increment xdot [%f, %f, %f, %f]\n", xdot_trajectory_d[state_dim * tid],
//                   xdot_trajectory_d[state_dim * tid + 1], xdot_trajectory_d[state_dim * tid + 2],
//                   xdot_trajectory_d[state_dim * tid + 3]);
//        }
    }
}

template<int STATE_DIM, int NUM_ROLLOUTS>
void launchIncrementStateAllRollouts_KernelTest(float dt, std::array<float, STATE_DIM*NUM_ROLLOUTS>& x_traj, std::array<float, STATE_DIM*NUM_ROLLOUTS>& xdot_traj) {
    // Declare variables for device memory
    float* x_traj_d;
    float* xdot_traj_d;

    // Allocate cuda memory
    HANDLE_ERROR(cudaMalloc((void**)&x_traj_d, sizeof(float)*x_traj.size()));
    HANDLE_ERROR(cudaMalloc((void**)&xdot_traj_d, sizeof(float)*xdot_traj.size()));


    // Copy the trajectories to the device
    HANDLE_ERROR(cudaMemcpy(x_traj_d, x_traj.data(), sizeof(float)*x_traj.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(xdot_traj_d, xdot_traj.data(), sizeof(float)*xdot_traj.size(), cudaMemcpyHostToDevice));

    dim3 blocksize(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 gridsize((NUM_ROLLOUTS + (BLOCKSIZE_X - 1)) / BLOCKSIZE_X, 1);

    // Launch the test kernel
    incrementStateAllRollouts_KernelTest<<<gridsize, blocksize>>>(STATE_DIM, NUM_ROLLOUTS, dt, x_traj_d, xdot_traj_d);
    CudaCheckError();

    // Copy the result back to the host
    HANDLE_ERROR(cudaMemcpy(x_traj.data(), x_traj_d, sizeof(float)*x_traj.size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(xdot_traj.data(), xdot_traj_d, sizeof(float)*xdot_traj.size(), cudaMemcpyDeviceToHost));

}

template<class COST_T>
__global__ void computeAndSaveCostAllRollouts_KernelTest(COST_T* cost, int state_dim, int num_rollouts, float* running_costs, float* terminal_state, float* cost_rollout_device) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x; // index on rollouts
//    if (tid == 0) {
//        printf("Current state [%f, %f, %f, %f]\n", terminal_state[state_dim * tid],
//               terminal_state[state_dim * tid + 1], terminal_state[state_dim * tid + 2],
//               terminal_state[state_dim * tid + 3]);
//        printf("Current cost [%f]\n", running_costs[tid]);
//    }
    mppi_common::computeAndSaveCost(num_rollouts, tid, cost, &terminal_state[state_dim*tid], running_costs[tid], cost_rollout_device);
//    if (tid == 0) {
//        printf("Total cost [%f]\n", cost_rollout_device[tid]);
//    }
}

template<class COST_T, int STATE_DIM, int NUM_ROLLOUTS>
void launchComputeAndSaveCostAllRollouts_KernelTest(COST_T cost,
        const std::array<float, NUM_ROLLOUTS>& cost_all_rollouts,
        const std::array<float, STATE_DIM*NUM_ROLLOUTS>& terminal_states,
        std::array<float, NUM_ROLLOUTS>& cost_compute) {
    // Allocate CUDA memory
    float* cost_all_rollouts_device;
    float* terminal_states_device;
    float* cost_compute_device;

    HANDLE_ERROR(cudaMalloc((void**)&cost_all_rollouts_device, sizeof(float)*cost_all_rollouts.size()));
    HANDLE_ERROR(cudaMalloc((void**)&terminal_states_device, sizeof(float)*terminal_states.size()));
    HANDLE_ERROR(cudaMalloc((void**)&cost_compute_device, sizeof(float)*cost_compute.size()));

    // Copy Host side data to the Device
    HANDLE_ERROR(cudaMemcpy(cost_all_rollouts_device, cost_all_rollouts.data(), sizeof(float)*cost_all_rollouts.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(terminal_states_device, terminal_states.data(), sizeof(float)*terminal_states.size(), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blocksize(BLOCKSIZE_X, 1);
    dim3 gridsize((NUM_ROLLOUTS + (BLOCKSIZE_X - 1)) / BLOCKSIZE_X, 1);
    computeAndSaveCostAllRollouts_KernelTest<<<blocksize, gridsize>>>(cost.cost_d_, STATE_DIM, NUM_ROLLOUTS, cost_all_rollouts_device, terminal_states_device, cost_compute_device);
    CudaCheckError();

    // Copy Device side data to the host
    HANDLE_ERROR(cudaMemcpy(cost_compute.data(), cost_compute_device, sizeof(float)*cost_compute.size(), cudaMemcpyDeviceToHost));

    // free cuda Memory
    cudaFree(cost_all_rollouts_device);
    cudaFree(terminal_states_device);
    cudaFree(cost_compute_device);
}


/***********************************************************************************************************************
 * Cartpole Running Cost Rollout Template Instantiations
 **********************************************************************************************************************/
const int num_timesteps_rc = 100;
const int num_rollouts_rc = 100;
template void computeRunningCostAllRollouts_CPU_TEST<CartpoleQuadraticCost, num_timesteps_rc, num_rollouts_rc, CartpoleDynamics::STATE_DIM, CartpoleDynamics::CONTROL_DIM>(
        CartpoleQuadraticCost& cost,
        float dt,
        std::array<float, CartpoleDynamics::STATE_DIM * num_timesteps_rc * num_rollouts_rc>& x_trajectory,
        std::array<float, CartpoleDynamics::CONTROL_DIM * num_timesteps_rc * num_rollouts_rc>& u_trajectory,
        std::array<float, CartpoleDynamics::CONTROL_DIM * num_timesteps_rc * num_rollouts_rc>& du_trajectory,
        std::array<float, CartpoleDynamics::CONTROL_DIM>& sigma_u,
        std::array<float, num_rollouts_rc>& cost_allrollouts);

template void launchComputeRunningCostAllRollouts_KernelTest<CartpoleQuadraticCost, num_timesteps_rc, num_rollouts_rc, CartpoleDynamics::STATE_DIM, CartpoleDynamics::CONTROL_DIM>(
        const CartpoleQuadraticCost& cost,
        float dt,
        const std::array<float, CartpoleDynamics::STATE_DIM * num_timesteps_rc * num_rollouts_rc>& x_trajectory,
        const std::array<float, CartpoleDynamics::CONTROL_DIM * num_timesteps_rc * num_rollouts_rc>& u_trajectory,
        const std::array<float, CartpoleDynamics::CONTROL_DIM * num_timesteps_rc * num_rollouts_rc>& du_trajectory,
        const std::array<float, CartpoleDynamics::CONTROL_DIM>& sigma_u,
        std::array<float, num_rollouts_rc>& cost_allrollouts);

/***********************************************************************************************************************
 * Cartpole Compute State Derivative Template Instantiations
 **********************************************************************************************************************/
 const int num_rollouts_sd = 1000;
template void launchComputeStateDerivAllRollouts_KernelTest<CartpoleDynamics, num_rollouts_sd>(const CartpoleDynamics& dynamics,
                                                   const std::array<float, CartpoleDynamics::STATE_DIM*num_rollouts_sd>& x_trajectory,
                                                   const std::array<float, CartpoleDynamics::CONTROL_DIM*num_rollouts_sd>& u_trajectory,
                                                   std::array<float, CartpoleDynamics::STATE_DIM*num_rollouts_sd>& xdot_trajectory);

/***********************************************************************************************************************
 * Cartpole Increment State Derivative Template Instantiations
 **********************************************************************************************************************/
const int num_rollouts_is = 5000;
template void launchIncrementStateAllRollouts_KernelTest<CartpoleDynamics::STATE_DIM, num_rollouts_is>(float dt,
        std::array<float, CartpoleDynamics::STATE_DIM*num_rollouts_is>& x_traj,
        std::array<float, CartpoleDynamics::STATE_DIM*num_rollouts_is>& xdot_traj);

/**
 * Cartpole Compute and Save cost all rollouts instantiations
 */
const int num_rollouts_cs = 1234;
template void launchComputeAndSaveCostAllRollouts_KernelTest<CartpoleQuadraticCost, CartpoleDynamics::STATE_DIM, num_rollouts_cs>(CartpoleQuadraticCost cost,
                                                    const std::array<float, num_rollouts_cs>& cost_all_rollouts,
                                                    const std::array<float, CartpoleDynamics::STATE_DIM*num_rollouts_cs>& terminal_states,
                                                    std::array<float, num_rollouts_cs>& cost_compute);
