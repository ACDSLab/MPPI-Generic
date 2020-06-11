/*
 * Created on Thu Jun 11 2020 by Bogdan
 */
#ifndef MPPI_COST_FUNCTIONS_QUADROTOR_QUADRATIC_COST_CUH_
#define MPPI_COST_FUNCTIONS_QUADROTOR_QUADRATIC_COST_CUH_

#include <mppi/cost_functions/cost.cuh>
struct QuadrotorQuadraticCostParams {
  float x_goal[3] = {0.0, 0.0, 0.0};
  float v_goal[3] = {0.0, 0.0, 0.0};
  // TODO Figure out angle goal
  // quat: w, x, y, z
  float q_goal[4] = {1, 0, 0, 0};
  float w_goal[3] = {0.0, 0.0, 0.0};
};

class QuadrotorQuadraticCost : public Cost<QuadrotorQuadraticCost,
                                           QuadrotorQuadraticCostParams, 13, 4> {
public:
  QuadrotorQuadraticCost(cudaStream_t stream = nullptr);
  ~QuadrotorQuadraticCost();

  /**
   * Host Functions
   */
  // void paramsToDevice();

  float computeStateCost(const Eigen::Ref<const state_array> s);

  float terminalCost(const Eigen::Ref<const state_array> s);

  /**
   * Devic Functions
   */
  __device__ float computeStateCost(float* s);

  __device__ float computeRunningCost(float* s, float* u, float* du, float* variance, int timestep);

  __device__ float terminalCost(float* s);
};

#if __CUDACC__
#include "quadrotor_quadratic_cost.cu"
#endif

#endif // MPPI_COST_FUNCTIONS_QUADROTOR_QUADRATIC_COSTS_CUH_
