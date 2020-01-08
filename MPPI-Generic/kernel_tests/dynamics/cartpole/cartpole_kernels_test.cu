//
// Created by mgandhi3 on 1/7/20.
//
#include <kernel_tests/dynamics/cartpole/cartpole_kernel_test.cuh>

__global__ void CartMassTestKernel(Cartpole* CP, float& mass_check) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("\nEntering the kernel!\n");
    // printf("The thread id is: %i\n", tid);
    if (tid == 0) {
        // printf("This is gravity: %f\n", CP->getGravity());
        // printf("This is the mass of the cart: %f\n", CP->getCartMass());
        // printf("This is the mass of the pole: %f\n", CP->getPoleMass());
        // printf("This is the length of the pole: %f\n", CP->getPoleLength());
        // printf("This is the value of GPUMemstatus on the GPU: %d\n", CP->GPUMemStatus_);
        // printf("This is the value of CP_device on the GPU: %d\n", CP->CP_device);
        mass_check = CP->getCartMass();
    }
}

__global__ void DyanmicsTestKernel(Cartpole* CP, float* state, float* control, float* state_der) {
    // int tid = blockIdx.x*blockDim.x + threadIdx.x;
    /**
     * This will probably do stupid things because of parallelization
     * Fix later
     */
    CP->xDot(state, control, state_der);
}

void launchCartMassTestKernel(const Cartpole& CP, float& mass_check) {
    // Allocate memory on the CPU for checking the mass
    float* mass_check_device;
    HANDLE_ERROR(cudaMalloc((void**)&mass_check_device, sizeof(float)));

    CartMassTestKernel<<<1,1>>>(CP.CP_device, *mass_check_device);
    CudaCheckError();

    // Copy the memory back to the host
    HANDLE_ERROR(cudaMemcpy(&mass_check, mass_check_device, sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(mass_check_device);
}

void launchDyanmicsTestKernel(const Cartpole& CP, float* state_cpu,
                              float* control_cpu, float* state_der_cpu) {
    // Allocate memory on the CPU for checking the mass
    float* state_gpu;
    float* control_gpu;
    float* state_der_gpu;

    HANDLE_ERROR(cudaMalloc((void**)&state_gpu, sizeof(float) * CP.STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&control_gpu, sizeof(float) * CP.CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&state_der_gpu, sizeof(float) * CP.STATE_DIM));

    HANDLE_ERROR(cudaMemcpy(state_gpu, state_cpu, sizeof(float)* CP.STATE_DIM, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(control_gpu, control_cpu, sizeof(float)* CP.CONTROL_DIM, cudaMemcpyHostToDevice));

    DyanmicsTestKernel<<<1,1>>>(CP.CP_device, state_gpu, control_gpu, state_der_gpu);
    CudaCheckError();

    // Copy the memory back to the host
    HANDLE_ERROR(cudaMemcpy(state_der_cpu, state_der_gpu, sizeof(float)* CP.STATE_DIM, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(state_gpu);
    cudaFree(control_gpu);
    cudaFree(state_der_gpu);
}
