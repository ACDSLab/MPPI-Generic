#pragma once
/*
Header file for costs
*/

#ifndef COSTS_CUH_
#define COSTS_CUH_

#include<Eigen/Dense>
#include <stdio.h>
#include <math.h>
#include <mppi/utils/managed.cuh>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdexcept>

/*
typedef struct {
  // fill in data here
} CostParams;
 */

// removing PARAMS_T is probably impossible
// https://cboard.cprogramming.com/cplusplus-programming/122412-crtp-how-pass-type.html
template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
class Cost : public Managed
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
     * typedefs for access to templated class from outside classes
     */
  static const int STATE_DIM = S_DIM;
  static const int CONTROL_DIM = C_DIM;
  typedef CLASS_T COST_T;
  typedef PARAMS_T COST_PARAMS_T;
  typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array; // Control at a time t
  typedef Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM> control_matrix; // Control at a time t
  typedef Eigen::Matrix<float, STATE_DIM, 1> state_array; // State at a time t

  Cost() = default;
  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a basePlant reference
   */
  virtual ~Cost() {
    freeCudaMem();
  }

  void GPUSetup();

  bool getDebugDisplayEnabled() {return false;}

  /**
   * returns a debug display that will be visualized based off of the state
   * @param state vector
   * @return
   */
  cv::Mat getDebugDisplay(float* s) {
    return cv::Mat();
  }

  /**
   * Updates the cost parameters
   * @param params
   */
  void setParams(PARAMS_T params) {
    params_ = params;
    if(GPUMemStatus_) {
      CLASS_T& derived = static_cast<CLASS_T&>(*this);
      derived.paramsToDevice();
    }
  }

  __host__ __device__ PARAMS_T getParams() {
    return params_;
  }

  void paramsToDevice();

  /**
   *
   * @param description
   * @param data
   */
  void updateCostmap(std::vector<int> description, std::vector<float> data) {};

  /**
   * deallocates the allocated cuda memory for an object
   */
  void freeCudaMem();

  /**
   * Computes the feedback control cost on CPU for RMPPI
   */
  float computeFeedbackCost(const Eigen::Ref<const control_array> fb_u,
                            const Eigen::Ref<const control_array> std_dev,
                            const float lambda = 1.0) {
    float cost = 0;
    for (int i = 0; i < CONTROL_DIM; i++) {
      cost += control_cost_coef[i] * fb_u(i) * fb_u(i) / powf(std_dev(i), 2);
    }

    return 0.5 * lambda * cost;
  }

  /**
   * Computes the control cost on CPU. This is the normal control cost calculation
   * in MPPI and Tube-MPPI
   */
  float computeLikelihoodRatioCost(const Eigen::Ref<const control_array> u,
                                   const Eigen::Ref<const control_array> noise,
                                   const Eigen::Ref<const control_array> std_dev,
                                   const float lambda = 1.0,
                                   const float alpha = 0.0) {
    float cost = 0;
    for (int i = 0; i < CONTROL_DIM; i++) {
      cost += control_cost_coef[i] * u(i) * (u(i) + 2 * noise(i)) /
        (std_dev(i) * std_dev(i));
    }
    return 0.5 * lambda * (1 - alpha) * cost;
  }

  /**
   * Computes the state cost on the CPU. Should be implemented in subclasses
   */
  float computeStateCost(const Eigen::Ref<const state_array> s) {
    throw std::logic_error("SubClass did not implement computeStateCost");
  }

  /**
   * Computes the state cost on the CPU. Should be implemented in subclasses
   */
  float terminalCost(const Eigen::Ref<const state_array> s) {
    throw std::logic_error("SubClass did not implement terminalCost");
  }

  /**
   * Computes the feedback control cost on GPU used in RMPPI. There is an
   * assumption that we are provided std_dev and the covriance matrix is
   * diagonal.
   */
  __device__ float computeFeedbackCost(float* fb_u, float* std_dev,
                                       float lambda = 1.0) {
    float cost = 0;
    for (int i = 0; i < CONTROL_DIM; i++) {
      cost += control_cost_coef[i] * powf(fb_u[i] / std_dev[i], 2);
    }
    return 0.5 * lambda * cost;
  }

  /**
   * Computes the normal control cost for MPPI and Tube-MPPI
   * 0.5 * lambda * (u*^T \Sigma^{-1} u*^T + 2 * u*^T \Sigma^{-1} (u*^T + noise))
   * On the GPU, u = u* + noise already, so we need the following to create
   * the original cost:
   * 0.5 * lambda * (u - noise)^T \Sigma^{-1} (u + noise)
   */
  __device__ float computeLikelihoodRatioCost(float* u,
                                              float* noise,
                                              float* std_dev,
                                              float lambda = 1.0,
                                              float alpha = 0.0) {
    float cost = 0;
    for (int i = 0; i < CONTROL_DIM; i++) {
      cost += control_cost_coef[i] * (u[i] - noise[i]) * (u[i] + noise[i]) /
        (std_dev[i] * std_dev[i]);
    }
    return 0.5 * lambda * (1 - alpha) * cost;
  }

  __device__ float computeStateCost(float* s);

  inline __host__ __device__ PARAMS_T getParams() const {return params_;}

  __device__ float computeRunningCost(float* s, float* u, float* du, float* std_dev, float lambda, float alpha, int timestep);
  __device__ float terminalCost(float* s);

  CLASS_T* cost_d_ = nullptr;
protected:
  PARAMS_T params_;
  // Not an Eigen control_array as it needs to exist on both CPU and GPU
  float control_cost_coef[CONTROL_DIM] = {1};
};

#if __CUDACC__
#include "cost.cu"
#endif

#endif // COSTS_CUH_
