#include <kernel_tests/mppi_core/rollout_kernel_test.cuh>

__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device,
        float* x_thread_device, float* xdot_thread_device, float* u_thread_device, float* du_thread_device, float* sigma_u_thread_device) {
    int thread_idx = threadIdx.x;
    int thread_idy = threadIdx.y;
    int block_idx = blockIdx.x;
    int global_idx = threadIdx.x + block_idx*blockDim.x;

    //Create shared arrays which hold state and control data
    __shared__ float x_shared[mppi_common::BLOCKSIZE_X * mppi_common::STATE_DIM];
    __shared__ float xdot_shared[mppi_common::BLOCKSIZE_X * mppi_common::STATE_DIM];
    __shared__ float u_shared[mppi_common::BLOCKSIZE_X * mppi_common::CONTROL_DIM];
    __shared__ float du_shared[mppi_common::BLOCKSIZE_X * mppi_common::CONTROL_DIM];
    __shared__ float sigma_u_shared[mppi_common::BLOCKSIZE_X * mppi_common::CONTROL_DIM];

    float* x_thread;
    float* xdot_thread;

    float* u_thread;
    float* du_thread;
    float* sigma_u_thread;

    if (global_idx < mppi_common::NUM_ROLLOUTS) {
        x_thread = &x_shared[thread_idx * mppi_common::STATE_DIM];
        xdot_thread = &xdot_shared[thread_idx * mppi_common::STATE_DIM];
        u_thread = &u_shared[thread_idx * mppi_common::CONTROL_DIM];
        du_thread = &du_shared[thread_idx * mppi_common::CONTROL_DIM];
        sigma_u_thread = &sigma_u_shared[thread_idx * mppi_common::CONTROL_DIM];
    }
    __syncthreads();
    mppi_common::loadGlobalToShared(mppi_common::STATE_DIM, mppi_common::CONTROL_DIM, mppi_common::NUM_ROLLOUTS,
            mppi_common::BLOCKSIZE_Y, global_idx, thread_idy,
            x0_device, sigma_u_device, x_thread, xdot_thread, u_thread, du_thread, sigma_u_thread);
    __syncthreads();

    // Check if on the first rollout the correct values were coped over
    if (global_idx == 1) {
        for (int i = 0; i < mppi_common::STATE_DIM; ++i) {
            x_thread_device[i] = x_thread[i];
            xdot_thread_device[i] = xdot_thread[i];
        }

        for (int i = 0; i < mppi_common::CONTROL_DIM; ++i) {
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
    HANDLE_ERROR(cudaMalloc((void**)&x0_device, sizeof(float)*mppi_common::STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&u_var_device, sizeof(float)*mppi_common::CONTROL_DIM));

    HANDLE_ERROR(cudaMemcpy(x0_device, x0_host.data(), sizeof(float)*mppi_common::STATE_DIM, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(u_var_device, u_var_host.data(), sizeof(float)*mppi_common::CONTROL_DIM, cudaMemcpyHostToDevice));


    // Define the return arguments in global device memory
    float* x_thread_device;
    float* xdot_thread_device;
    float* u_thread_device;
    float* du_thread_device;
    float* sigma_u_thread_device;

    HANDLE_ERROR(cudaMalloc((void**)&x_thread_device, sizeof(float)*mppi_common::STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&xdot_thread_device, sizeof(float)*mppi_common::STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&u_thread_device, sizeof(float)*mppi_common::CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&du_thread_device, sizeof(float)*mppi_common::CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&sigma_u_thread_device, sizeof(float)*mppi_common::CONTROL_DIM));

    dim3 dimBlock(mppi_common::BLOCKSIZE_X, mppi_common::BLOCKSIZE_Y);
    dim3 dimGrid(2048);

    loadGlobalToShared_KernelTest<<<dimGrid,dimBlock>>>(x0_device, u_var_device,
            x_thread_device, xdot_thread_device, u_thread_device, du_thread_device, sigma_u_thread_device);
    CudaCheckError();

    // Copy the data back to the host
    HANDLE_ERROR(cudaMemcpy(x_thread_host.data(), x_thread_device, sizeof(float)*mppi_common::STATE_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(xdot_thread_host.data(), xdot_thread_device, sizeof(float)*mppi_common::STATE_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u_thread_host.data(), u_thread_device, sizeof(float)*mppi_common::CONTROL_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(du_thread_host.data(), du_thread_device, sizeof(float)*mppi_common::CONTROL_DIM, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(sigma_u_thread_host.data(), sigma_u_thread_device, sizeof(float)*mppi_common::CONTROL_DIM, cudaMemcpyDeviceToHost));

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
    float u_thread[mppi_common::CONTROL_DIM];
    float du_thread[mppi_common::CONTROL_DIM];

    if (global_idx < num_rollouts) {
        mppi_common::injectControlNoise(mppi_common::CONTROL_DIM, mppi_common::BLOCKSIZE_Y, num_rollouts, num_timesteps, timestep, global_idx, thread_idy, u_traj_device, ep_v_device, sigma_u_device, u_thread, du_thread);
        if (thread_idy < mppi_common::CONTROL_DIM) {
            control_compute_device[global_idx * mppi_common::CONTROL_DIM + thread_idy] = u_thread[thread_idy];
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
    dim3 block_size(mppi_common::BLOCKSIZE_X, mppi_common::BLOCKSIZE_Y);
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