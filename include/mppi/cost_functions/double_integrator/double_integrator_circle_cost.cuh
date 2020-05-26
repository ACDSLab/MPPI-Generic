#pragma once
#ifndef DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_
#define DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_

#include <mppi/cost_functions/cost.cuh>

typedef struct {
  float velocity_cost = 1;
  float crash_cost = 1000;
  float velocity_desired = 2;
  float inner_path_radius2 = 1.875*1.875;
  float outer_path_radius2 = 2.125*2.125;
  float angular_momentum_desired = 2*velocity_desired; // Enforces the system travels counter clockwise
} DoubleIntegratorCircleCostParams;

class DoubleIntegratorCircleCost : public Cost<DoubleIntegratorCircleCost,
                                               DoubleIntegratorCircleCostParams, 4, 2> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DoubleIntegratorCircleCost(cudaStream_t stream = nullptr);

  ~DoubleIntegratorCircleCost();

  void paramsToDevice();
  float computeStateCost(const Eigen::Ref<const state_array> s);
  float terminalCost(const Eigen::Ref<const state_array> s);

  __device__ float getControlCost(float* u, float* du, float* variance);

  __device__ float getStateCost(float* s);

  __device__ float computeRunningCost(float* s, float* u, float* du, float* variance, int timestep);

  __device__ float terminalCost(float* s);


};

#if __CUDACC__
#include "double_integrator_circle_cost.cu"
#endif


#endif //!DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_