#ifndef DOUBLE_INTEGRATOR_ROBUST_COST_CUH_
#define DOUBLE_INTEGRATOR_ROBUST_COST_CUH_

#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>

class DoubleIntegratorRobustCost : public DoubleIntegratorCircleCost {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DoubleIntegratorRobustCost(cudaStream_t stream = nullptr);

  ~DoubleIntegratorRobustCost();

  float computeStateCost(const Eigen::Ref<const state_array> s);
  float terminalCost(const Eigen::Ref<const state_array> s);

  __device__ float getControlCost(float* u, float* du, float* variance);

  __device__ float computeStateCost(float* s);

  __device__ float computeRunningCost(float* s, float* u, float* du, float* variance, int timestep);

  __device__ float terminalCost(float* s);


};

#if __CUDACC__
#include "double_integrator_robust_cost.cu"
#endif


#endif //!DOUBLE_INTEGRATOR_ROBUST_COST_CUH_