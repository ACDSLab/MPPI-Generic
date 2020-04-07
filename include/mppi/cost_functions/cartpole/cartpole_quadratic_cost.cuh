#pragma once

#ifndef CARTPOLE_QUADRATIC_COST_CUH_
#define CARTPOLE_QUADRATIC_COST_CUH_

#include <mppi/cost_functions/cost.cuh>
#include <mppi/utils/file_utils.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cuda_runtime.h>

typedef struct {
  float cart_position_coeff = 1000;
  float cart_velocity_coeff = 100;
  float pole_angle_coeff = 2000;
  float pole_angular_velocity_coeff = 100;
  float control_force_coeff = 1;
  float terminal_cost_coeff = 0;
  float desired_terminal_state[4] = {0, 0, M_PI, 0};
} cartpoleQuadraticCostParams;

class CartpoleQuadraticCost : public Cost<CartpoleQuadraticCost, cartpoleQuadraticCostParams> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * Constructor
   * @param width
   * @param height
   */
  CartpoleQuadraticCost(cudaStream_t stream=0);

  /**
   *
   */
  ~CartpoleQuadraticCost();


  /**
   * Copies the parameters to the GPU object
   */
  void paramsToDevice();

  /**
   * @brief Compute the control cost
   */
  __host__ __device__ float getControlCost(float* u, float* du, float* vars);

  /**
   * @brief Compute the state cost
   */
  __host__ __device__ float getStateCost(float* s);


  /**
   * @brief Compute all of the individual cost terms and adds them together.
   */
  __host__ __device__ float computeRunningCost(float* s, float* u, float* du, float* vars, int timestep);

  /**
   * @brief Compute the terminal cost of the system
   */
   __host__ __device__ float terminalCost(float *s);

protected:

};



#if __CUDACC__
#include "cartpole_quadratic_cost.cu"
#endif

#endif // CARTPOLE_QUADRATIC_COST_CUH_// Include the cart pole cost.
