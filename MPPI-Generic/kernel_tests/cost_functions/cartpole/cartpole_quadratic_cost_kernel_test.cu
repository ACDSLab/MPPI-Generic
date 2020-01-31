#include "cartpole_quadratic_cost_kernel_test.cuh"

__global__ void parameterTestKernel(CartpoleQuadraticCost* cost_d, CartpoleQuadraticCost::Params& params_d) {
    // The parameters have been set outside of the kernel on the device, copy the current values of the parameters to params_d
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid == 0) {
        params_d = cost_d->getParams();
    }
}

void launchParameterTestKernel(const CartpoleQuadraticCost& cost, CartpoleQuadraticCost::Params& param_check) {
    // Allocate memory for the device side parameter structure
    CartpoleQuadraticCost::Params* param_d = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&param_d, sizeof(CartpoleQuadraticCost::Params)))

    parameterTestKernel<<<1,1>>>(cost.cost_d_, *param_d);
    CudaCheckError();

    HANDLE_ERROR(cudaMemcpy(&param_check, param_d, sizeof(param_check), cudaMemcpyDeviceToHost))

    cudaFree(param_d);
}


