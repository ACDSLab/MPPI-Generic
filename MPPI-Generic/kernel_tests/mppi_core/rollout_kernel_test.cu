#include <kernel_tests/mppi_core/rollout_kernel_test.cuh>

__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device) {
    int thread_idx = threadIdx.x;
    int thread_idy = threadIdx.y;
    int block_idx = blockIdx.x;
    int global_idx = threadIdx.x + block_idx*blockDim.x;


    // Declare variables that are local to the thread
    float* x_thread;
    float* xdot_thread;
    float* u_thread;
    float* du_thread;
    float* sigma_u_thread;

    //Create shared arrays which hold state and control data
    __shared__ float x_shared[mppi_common::blocksize_x*mppi_common::state_dim];
    __shared__ float xdot_shared[mppi_common::blocksize_x*mppi_common::state_dim];
    __shared__ float u_shared[mppi_common::blocksize_x*mppi_common::control_dim];
    __shared__ float du_shared[mppi_common::blocksize_x*mppi_common::control_dim];
    __shared__ float sigma_u_shared[mppi_common::blocksize_x*mppi_common::control_dim];

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
}

void launchGlobalToShared_KernelTest() {

    // Define the initial condition x0_device and the exploration variance in global device memory
    float* x0_device;
    float* u_var_device;
    HANDLE_ERROR(cudaMalloc((void**)&x0_device, sizeof(float)*mppi_common::state_dim));
    HANDLE_ERROR(cudaMalloc((void**)&u_var_device, sizeof(float)*mppi_common::control_dim));


    //loadGlobalToShared_KernelTest<<<1,1>>>();
    CudaCheckError();
}