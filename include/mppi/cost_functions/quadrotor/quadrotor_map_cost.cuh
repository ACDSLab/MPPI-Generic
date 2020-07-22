/*
 * Created on Wed Jul 22 2020 by Bogdan
 */
#ifndef MPPI_COST_FUNCTIONS_QUADROTOR_MAP_COST_CUH_
#define MPPI_COST_FUNCTIONS_QUADROTOR_MAP_COST_CUH_

#include <mppi/cost_functions/cost.cuh>
#include <mppi/utils/math_utils.h>

struct QuadrotorMapCostParams : public CostParams<4> {
  QuadrotorMapCostParams() {
    control_cost_coeff[0] = 1;  // roll
    control_cost_coeff[1]] = 1; // pitch
    control_cost_coeff[2] = 1;  // yaw
    control_cost_coeff[3] = 1;  // throttle
  }
}

template <class CLASS_T, class PARAMS_T = QuadrotorMapCostParams>
class QuadrotorMapCostImpl : public Cost<CLASS_T, PARAMS_T, 13, 4> {
public:
  QuadrotorMapCostImpl(cudaStream_t stream = 0);

  std::string getCostFunctionName() {return "Quadrotor Map Cost";}

  float computeStateCost(const Eigen::ref<const state_array> s, int timestep,
                         int* crash_status);

  __device__ float computeStateCost(float* s, int timestep, int* crash_status);

  float terminalCost(const Eigen::Ref<const state_array> s);

  __device__ float terminalCost(float* s);
protected:
  std::string map_path_;
}

class QuadrotorMapCost : public QuadrotorMapCostImpl<QuadrotorMapCost> {
public:
  QuadrotorMapCost(cudaStream_t stream = 0) : QuadrotorMapCostImpl<QuadrotorMapCost>(stream) {};
}

#if __CUDACC__
#include "quadrotor_map_cost.cu"
#endif
#endif  // MPPI_COST_FUNCTIONS_QUADROTOR_MAP_COST_CUH_