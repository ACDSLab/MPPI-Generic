#pragma once
#ifndef DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_
#define DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_

#include <cost_functions/cost.cuh>

typedef struct {
  float velocity_cost = 1;
  float crash_cost = 1000;
  float velocity_desired = 1;
  float inner_path_radius2 = 1.875*1.875;
  float outer_path_radius2 = 2.125*2.125;
} DoubleIntegratorCircleCostParams;

class DoubleIntegratorCircleCost : public Cost<DoubleIntegratorCircleCost, DoubleIntegratorCircleCostParams> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DoubleIntegratorCircleCost(cudaStream_t stream = nullptr);

  ~DoubleIntegratorCircleCost();

  void paramsToDevice();

  __host__ __device__ float getControlCost(float* u, float* du, float* variance);

  __host__ __device__ float getStateCost(float* s);

  __host__ __device__ float computeRunningCost(float* s, float* u, float* du, float* variance, int timestep);

  __host__ __device__ float terminalCost(float* s);


};

#if __CUDACC__
#include "double_integrator_circle_cost.cu"
#endif


#endif //!DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_