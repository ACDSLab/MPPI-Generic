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

  float s_goal[13] = {0, 0, 0,
                      0, 0, 0,
                      1, 0, 0, 0,
                      0, 0, 0};

  float x_coeff = 1.0;
  float v_coeff = 1.0;
  float q_coeff = 1.0;
  float w_coeff = 1.0;
  float terminal_cost_coeff = 0;

  Eigen::Matrix<float, 13, 1> getDesiredState() {
    Eigen::Matrix<float, 13, 1> s;
    s << x_goal[0], x_goal[1], x_goal[2],
         v_goal[0], v_goal[1], v_goal[2],
         q_goal[0], q_goal[1], q_goal[2], q_goal[3],
         w_goal[0], w_goal[1], w_goal[2];
    return s;
  }
};

class QuadrotorQuadraticCost : public Cost<QuadrotorQuadraticCost,
                                           QuadrotorQuadraticCostParams, 13, 4> {
public:
  QuadrotorQuadraticCost(cudaStream_t stream = nullptr);

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
