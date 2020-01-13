#include <kernel_tests/mppi_core/rollout_kernel_test.cuh>

__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device,
        float* x_thread_device, float* xdot_thread_device, float* u_thread_device, float* du_thread_device, float* sigma_u_thread_device) {
    int thread_idx = threadIdx.x;
    int thread_idy = threadIdx.y;
    int block_idx = blockIdx.x;
    int global_idx = threadIdx.x + block_idx*blockDim.x;

    //Create shared arrays which hold state and control data
    __shared__ float x_shared[mppi_common::blocksize_x*mppi_common::state_dim];
    __shared__ float xdot_shared[mppi_common::blocksize_x*mppi_common::state_dim];
    __shared__ float u_shared[mppi_common::blocksize_x*mppi_common::control_dim];
    __shared__ float du_shared[mppi_common::blocksize_x*mppi_common::control_dim];
    __shared__ float sigma_u_shared[mppi_common::blocksize_x*mppi_common::control_dim];

    float* x_thread;
    float* xdot_thread;

    float* u_thread;
    float* du_thread;
    float* sigma_u_thread;

    if (global_idx < mppi_common::num_rollouts) {
        x_thread = &x_shared[thread_idx * mppi_common::state_dim];
        xdot_thread = &xdot_shared[thread_idx * mppi_common::state_dim];
        u_thread = &u_shared[thread_idx * mppi_common::control_dim];
        du_thread = &du_shared[thread_idx * mppi_common::control_dim];
        sigma_u_thread = &sigma_u_shared[thread_idx * mppi_common::control_dim];
    }
    __syncthreads();
    mppi_common::loadGlobalToShared(global_idx, thread_idy, x0_device, sigma_u_device, x_thread,
                                    xdot_thread, u_thread, du_thread, sigma_u_thread);
    __syncthreads();

    // Check if on the first rollout the correct values were coped over
    if (global_idx == 1) {
        for (int i = 0; i < mppi_common::state_dim; ++i) {
            x_thread_device[i] = x_thread[i];
            xdot_thread_device[i] = xdot_thread[i];
        }

        for (int i = 0; i < mppi_common::control_dim; ++i) {
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
    HANDLE_ERROR(cudaMalloc((void**)&x0_device, sizeof(float)*mppi_common::state_dim));
    HANDLE_ERROR(cudaMalloc((void**)&u_var_device, sizeof(float)*mppi_common::control_dim));

    HANDLE_ERROR(cudaMemcpy(x0_device, x0_host.data(), sizeof(float)*mppi_common::state_dim, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(u_var_device, u_var_host.data(), sizeof(float)*mppi_common::control_dim, cudaMemcpyHostToDevice));


    // Define the return arguments in global device memory
    float* x_thread_device;
    float* xdot_thread_device;
    float* u_thread_device;
    float* du_thread_device;
    float* sigma_u_thread_device;

    HANDLE_ERROR(cudaMalloc((void**)&x_thread_device, sizeof(float)*mppi_common::state_dim));
    HANDLE_ERROR(cudaMalloc((void**)&xdot_thread_device, sizeof(float)*mppi_common::state_dim));
    HANDLE_ERROR(cudaMalloc((void**)&u_thread_device, sizeof(float)*mppi_common::control_dim));
    HANDLE_ERROR(cudaMalloc((void**)&du_thread_device, sizeof(float)*mppi_common::control_dim));
    HANDLE_ERROR(cudaMalloc((void**)&sigma_u_thread_device, sizeof(float)*mppi_common::control_dim));

    dim3 dimBlock(mppi_common::blocksize_x, mppi_common::blocksize_y);
    dim3 dimGrid(2048);

    loadGlobalToShared_KernelTest<<<dimGrid,dimBlock>>>(x0_device, u_var_device,
            x_thread_device, xdot_thread_device, u_thread_device, du_thread_device, sigma_u_thread_device);
    CudaCheckError();

    // Copy the data back to the host
    HANDLE_ERROR(cudaMemcpy(x_thread_host.data(), x_thread_device, sizeof(float)*mppi_common::state_dim, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(xdot_thread_host.data(), xdot_thread_device, sizeof(float)*mppi_common::state_dim, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u_thread_host.data(), u_thread_device, sizeof(float)*mppi_common::control_dim, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(du_thread_host.data(), du_thread_device, sizeof(float)*mppi_common::control_dim, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(sigma_u_thread_host.data(), sigma_u_thread_device, sizeof(float)*mppi_common::control_dim, cudaMemcpyDeviceToHost));

    // Free the cuda memory that we allocated
    cudaFree(x0_device);
    cudaFree(u_var_device);

    cudaFree(x_thread_device);
    cudaFree(xdot_thread_device);
    cudaFree(u_thread_device);
    cudaFree(du_thread_device);
    cudaFree(sigma_u_thread_device);
}