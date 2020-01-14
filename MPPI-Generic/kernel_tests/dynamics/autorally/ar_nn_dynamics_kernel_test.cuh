#ifndef AR_NN_DYNAMICS_KERNEL_TEST_CUH
#define AR_NN_DYNAMICS_KERNEL_TEST_CUH

#include <array>
#include <dynamics/autorally/ar_nn_model.cuh>
#include <cuda_runtime.h>

template<class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
void launchParameterCheckTestKernel(NETWORK_T& model, std::array<float, THETA_SIZE>& theta, std::array<int, STRIDE_SIZE>& stride,
        std::array<int, NUM_LAYERS>& net_structure);


template<class NETWORK_T>
__global__ void parameterCheckTestKernel(NETWORK_T* model,  float* theta, int* stride, int* net_structure);


#endif //AR_NN_DYNAMICS_KERNEL_TEST_CUH
