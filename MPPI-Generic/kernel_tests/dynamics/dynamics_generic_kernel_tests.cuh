#ifndef DYNAMICS_GENERIC_KERNEL_TESTS_CUH_
#define DYNAMICS_GENERIC_KERNEL_TESTS_CUH_

#include <array>
#include <vector>

template<typename CLASS_T, typename PARAMS_T>
__global__ void parameterTestKernel(CLASS_T* class_t, PARAMS_T& params);
template<typename CLASS_T, typename PARAMS_T>
void launchParameterTestKernel(CLASS_T& class_t, PARAMS_T& params);

template<typename DYNAMICS_T, typename PARAMS_T, int C_DIM>
__global__ void controlRangesTestKernel( DYNAMICS_T* dynamics, float2* control_rngs);
template<typename DYNAMICS_T, typename PARAMS_T, int C_DIM>
void launchControlRangesTestKernel( DYNAMICS_T& dynamics, std::array<float2, C_DIM>& control_rngs);

template<typename DYNAMICS_T, typename PARAMS_T, int S_DIM, int C_DIM>
__global__ void enforceConstraintTestKernel( DYNAMICS_T* dynamics, float* state, float* control, int num);
template<typename DYNAMICS_T, typename PARAMS_T, int S_DIM, int C_DIM>
void launchEnforceConstraintTestKernel( DYNAMICS_T& dynamics, std::vector<std::array<float, S_DIM>>& state,
        std::vector< std::array<float, C_DIM>>& control, int dim_y);

template<typename DYNAMICS_T, typename PARAMS_T, int S_DIM>
__global__ void updateStateTestKernel( DYNAMICS_T* dynamics, float* state, float* state_der, float dt, int num);
template<typename DYNAMICS_T, typename PARAMS_T, int S_DIM>
void launchUpdateStateTestKernel( DYNAMICS_T& dynamics, std::vector< std::array<float, S_DIM>>& state,
        std::vector< std::array<float, S_DIM>>& state_der, float dt, int dim_y);

template<typename DYNAMICS_T, typename PARAMS_T, int S_DIM, int C_DIM>
__global__ void computeStateDerivTestKernel( DYNAMICS_T* dynamics, float* state, float* control, float* state_der, int num);
template<typename DYNAMICS_T, typename PARAMS_T, int S_DIM, int C_DIM>
void launchComputeStateDerivTestKernel( DYNAMICS_T& dynamics, std::vector< std::array<float, S_DIM>>& state,
        std::vector< std::array<float, C_DIM>>& control, std::vector< std::array<float, S_DIM>>& state_der, int dim_y);


#if __CUDACC__
#include "dynamics_generic_kernel_tests.cu"
#endif

#endif // DYNAMICS_GENERIC_KERNEL_TESTS_CUH_
