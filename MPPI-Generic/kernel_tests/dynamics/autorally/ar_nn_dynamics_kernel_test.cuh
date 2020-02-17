#ifndef AR_NN_DYNAMICS_KERNEL_TEST_CUH
#define AR_NN_DYNAMICS_KERNEL_TEST_CUH

#include <array>
#include <dynamics/autorally/ar_nn_model.cuh>
#include <cuda_runtime.h>

template<class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
void launchParameterCheckTestKernel(NETWORK_T& model, std::array<float, THETA_SIZE>& theta, std::array<int, STRIDE_SIZE>& stride,
        std::array<int, NUM_LAYERS>& net_structure);


template <class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
__global__ void parameterCheckTestKernel(NETWORK_T* model,  float* theta, int* stride, int* net_structure);


template<class NETWORK_T, int BLOCK_DIM_Y, int STATE_DIM>
void launchIncrementStateTestKernel(NETWORK_T& model, std::array<float, STATE_DIM>& state, std::array<float, 7>& state_der);

template<class NETWORK_T, int STATE_DIM>
__global__ void incrementStateTestKernel(NETWORK_T* model, float* state, float* state_der);

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM, int BLOCKSIZE_Y>
void launchComputeDynamicsTestKernel(NETWORK_T& model, float* state, float* control, float* state_der);

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM>
__global__ void computeDynamicsTestKernel(NETWORK_T* model, float* state, float* control, float* state_der);

template<class NETWORK_T, int STATE_DIM, int CONTORL_DIM, int BLOCKSIZE_Y>
void launchComputeStateDerivTestKernel(NETWORK_T& model, float* state, float* control, float* state_der);

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM>
__global__ void computeStateDerivTestKernel(NETWORK_T* model, float* state, float* control, float* state_der);

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM, int BLOCKSIZE_Y>
void launchFullARNNTestKernel(NETWORK_T& model, float* state, float* control, float* state_der);

template<class NETWORK_T, int STATE_DIM, int CONTROL_DIM>
__global__ void fullARNNTestKernel(NETWORK_T* model, float* state, float* control, float* state_der);

#include "ar_nn_dynamics_kernel_test.cu"

#endif //AR_NN_DYNAMICS_KERNEL_TEST_CUH
