#include "weightedreduction_kernel_test.cuh"

__global__ void setInitialControlToZero_KernelTest(int control_dim, float* u_d, float* u_intermediate) {
    int thread_idx = blockDim.x*blockIdx.x + threadIdx.x;
    mppi_common::setInitialControlToZero(control_dim, thread_idx, u_d, u_intermediate);
}

template<int num_threads, int control_dim>
void launchSetInitialControlToZero_KernelTest(std::array<float, control_dim>& u_host,
                                              std::array<float, num_threads*control_dim>& u_intermediate_host) {

    float* u_dev;
    float* u_intermediate_dev;

    // Allocate Memory
    HANDLE_ERROR(cudaMalloc((void**)&u_dev, sizeof(float)*u_host.size()))
    HANDLE_ERROR(cudaMalloc((void**)&u_intermediate_dev, sizeof(float)*u_intermediate_host.size()))

    setInitialControlToZero_KernelTest<<<1,num_threads>>>(control_dim, u_dev, u_intermediate_dev);
    CudaCheckError();

    HANDLE_ERROR(cudaMemcpy(u_host.data(), u_dev, sizeof(float)*u_host.size(), cudaMemcpyDeviceToHost))
    HANDLE_ERROR(cudaMemcpy(u_intermediate_host.data(), u_intermediate_dev, sizeof(float)*u_intermediate_host.size(), cudaMemcpyDeviceToHost))

    cudaFree(u_dev);
    cudaFree(u_intermediate_dev);
}

template<int control_dim>
__global__ void strideControlWeightReduction_KernelTest(int num_rollouts, int num_timesteps, int sum_stride,
        float* exp_costs_d, float normalizer, float* du_d, float* u_intermediate) {
    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;

    float u_thread[control_dim];
    float* u_intermediate_thread = &u_intermediate[block_idx*control_dim*((num_rollouts - 1) / sum_stride + 1)];

    mppi_common::strideControlWeightReduction(num_rollouts, num_timesteps, sum_stride,
            thread_idx, block_idx, control_dim, exp_costs_d, normalizer, du_d,
                                              u_thread, u_intermediate_thread);
}

template<int control_dim, int num_rollouts, int num_timesteps, int sum_stride>
void launchStrideControlWeightReduction_KernelTest(float normalizer,
                                                   const std::array<float, num_rollouts>& exp_costs_host,
                                                   const std::array<float, num_rollouts*num_timesteps*control_dim>& du_host,
                                                   std::array<float, num_timesteps*control_dim*((num_rollouts - 1) / sum_stride + 1)>& u_intermediate_host) {
    float* exp_costs_dev;
    float* du_dev;
    float* u_intermediate_dev;

    // Allocate Memory
    HANDLE_ERROR(cudaMalloc((void**)&exp_costs_dev, sizeof(float)*exp_costs_host.size()));
    HANDLE_ERROR(cudaMalloc((void**)&du_dev, sizeof(float)*du_host.size()))
    HANDLE_ERROR(cudaMalloc((void**)&u_intermediate_dev, sizeof(float)*u_intermediate_host.size()));

    HANDLE_ERROR(cudaMemcpy(exp_costs_dev, exp_costs_host.data(), sizeof(float)*exp_costs_host.size(), cudaMemcpyHostToDevice))
    HANDLE_ERROR(cudaMemcpy(du_dev, du_host.data(), sizeof(float)*du_host.size(), cudaMemcpyHostToDevice))

    dim3 blockdim((num_rollouts - 1) / sum_stride + 1, 1, 1);
    dim3 griddim(num_timesteps, 1, 1);

    strideControlWeightReduction_KernelTest<control_dim><<<griddim,blockdim>>>(num_rollouts, num_timesteps, sum_stride,
    exp_costs_dev, normalizer, du_dev, u_intermediate_dev);
    CudaCheckError();

    HANDLE_ERROR(cudaMemcpy(u_intermediate_host.data(), u_intermediate_dev, sizeof(float)*u_intermediate_host.size(), cudaMemcpyDeviceToHost))

    cudaFree(exp_costs_dev);
    cudaFree(du_dev);
    cudaFree(u_intermediate_dev);
}

template<int control_dim>
__global__ void rolloutWeightReductionAndSaveControl_KernelTest(int num_rollouts, int num_timesteps, int sum_stride,
        float* u_intermediate, float* du_new_d) {
    int thread_idx = threadIdx.x; // Current cell
    int block_idx = blockIdx.x; // Current timestep

    float u[control_dim];
    float* u_intermediate_thread = &u_intermediate[block_idx*control_dim*((num_rollouts - 1) / sum_stride + 1)];
    mppi_common::rolloutWeightReductionAndSaveControl(thread_idx, block_idx, num_rollouts, num_timesteps, control_dim, sum_stride,
            u, u_intermediate_thread, du_new_d);
}

template<int control_dim, int num_rollouts, int num_timesteps, int sum_stride>
void launchRolloutWeightReductionAndSaveControl_KernelTest(const std::array<float, num_timesteps*control_dim*((num_rollouts - 1) / sum_stride + 1)>& u_intermediate_host,
        std::array<float, num_timesteps*control_dim>& du_new_host) {
    float* u_intermediate_dev;
    float* du_new_dev;

    // Allocate Memory
    HANDLE_ERROR(cudaMalloc((void**)&u_intermediate_dev, sizeof(float)*u_intermediate_host.size()))
    HANDLE_ERROR(cudaMalloc((void**)&du_new_dev, sizeof(float)*du_new_host.size()))

    HANDLE_ERROR(cudaMemcpy(u_intermediate_dev, u_intermediate_host.data(), sizeof(float)*u_intermediate_host.size(), cudaMemcpyHostToDevice))
    dim3 blockdim((num_rollouts - 1) / sum_stride + 1, 1, 1);
    dim3 griddim(num_timesteps, 1, 1);

    rolloutWeightReductionAndSaveControl_KernelTest<control_dim><<<griddim, blockdim>>>(num_rollouts, num_timesteps,
            sum_stride, u_intermediate_dev, du_new_dev);
    CudaCheckError();

    HANDLE_ERROR(cudaMemcpy(du_new_host.data(), du_new_dev, sizeof(float)*du_new_host.size(), cudaMemcpyDeviceToHost))

    cudaFree(u_intermediate_dev);
    cudaFree(du_new_dev);
}
