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
  virtual ~Cost() = default;

  void GPUSetup() {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    if (!GPUMemStatus_) {
      this->cost_d_ = Managed::GPUSetup(derived);
    } else {
      std::cout << "GPU Memory already set" << std::endl; //TODO should this be an exception?
    }
    derived->paramsToDevice();
  }

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
    this->params_ = params;
    if(this->GPUMemStatus_) {
      CLASS_T& derived = static_cast<CLASS_T&>(*this);
      derived.paramsToDevice();
    }
  }

  __host__ __device__ PARAMS_T getParams() {
    return params_;
  }

  void paramsToDevice() {
    printf("ERROR: calling paramsToDevice of base cost");
    exit(1);
  };

  /**
   *
   * @param description
   * @param data
   */
  void updateCostmap(std::vector<int> description, std::vector<float> data) {};

  /**
   * deallocates the allocated cuda memory for an object
   */
  void freeCudaMem() {
    if(GPUMemStatus_) {
      cudaFree(cost_d_);
      this->GPUMemStatus_ = false;
      cost_d_ = nullptr;
    }
  }

  /**
   * Computes the control cost on CPU
   */
  float computeFeeedbackCost(const Eigen::Ref<const control_array> ff_u,
                             const Eigen::Ref<const control_array> noise,
                             const Eigen::Ref<const control_array> fb_u,
                             const Eigen::Ref<const control_matrix> variance) {
    control_array u = ff_u + noise + fb_u;
    return u.transpose() * variance.inverse() * u;
  }

  /**
   * Computes the state cost on the CPU. Should be implemented in subclasses
   */
  float computeStateCost(const Eigen::Ref<const state_array> s) {
    throw std::logic_error("SubClass did not implement computeStateCost");
  }

  /**
   * Computes the control cost on GPU. There is an assumption that we are
   * provided std_dev and the covriance matrix is diagonal
   */
  __device__ float computeFeedbackCost(float* u, float* noise, float* fb_u,
                                       float* std_dev) {
    float cost = 0;
    float tmp_var = 0;
    for (int i = 0; i < CONTROL_DIM; i++) {
      tmp_var = ((u[i] + noise[i] + fb_u[i]) / std_dev[i]);
      cost += tmp_var * tmp_var;
    }
    return cost;
  }

  inline __host__ __device__ PARAMS_T getParams() const {return this->params_;}

  __device__ float computeRunningCost(float* s, float* u, float* du, float* vars, int timestep);
  __device__ float terminalCost(float* s);

  CLASS_T* cost_d_ = nullptr;
protected:
  PARAMS_T params_;

};

#endif // COSTS_CUH_
