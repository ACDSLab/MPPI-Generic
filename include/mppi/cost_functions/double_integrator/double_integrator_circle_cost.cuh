#pragma once
#ifndef DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_
#define DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_

#include <mppi/cost_functions/cost.cuh>


struct DoubleIntegratorCircleCostParams : public CostParams<2> {
  float velocity_cost = 10;
  float crash_cost = 10000;
  float velocity_desired = 10;
  float inner_path_radius2 = 1.875*1.875;
  float outer_path_radius2 = 2.125*2.125;
  float angular_momentum_desired = 2*velocity_desired; // Enforces the system travels counter clockwise

  DoubleIntegratorCircleCostParams() {
    control_cost_coeff[0] = 0.01;
    control_cost_coeff[1] = 0.01;
    discount = 1.0;
  }
} ;

class DoubleIntegratorCircleCost : public Cost<DoubleIntegratorCircleCost,
                                               DoubleIntegratorCircleCostParams, 4, 2> {
public:
  DoubleIntegratorCircleCost(cudaStream_t stream = nullptr);

  float computeStateCost(const Eigen::Ref<const state_array> s, int timestep=0, int* crash_status=nullptr);
  float terminalCost(const Eigen::Ref<const state_array> s);

  __device__ float computeStateCost(float* s, int timestep=0, int* crash_status=nullptr);
  __device__ float terminalCost(float* s);
};

#if __CUDACC__
#include "double_integrator_circle_cost.cu"
#endif


#endif //!DOUBLE_INTEGRATOR_CIRCLE_COST_CUH_
