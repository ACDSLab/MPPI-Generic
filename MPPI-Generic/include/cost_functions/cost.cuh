#pragma once
/*
Header file for costs
*/

#ifndef COSTS_CUH_
#define COSTS_CUH_

#include<Eigen/Dense>
#include <stdio.h>
#include <math.h>
#include <utils/managed.cuh>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
typedef struct {
  // fill in data here
} CostParams;
 */

template<class CLASS_T, class PARAMS_T>
class Cost : public Managed
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
     * typedefs for access to templated class from outside classes
     */
  typedef CLASS_T COST_T;
  typedef PARAMS_T COST_PARAMS_T;

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

  inline __host__ __device__ PARAMS_T getParams() const {return this->params_;}

  __device__ float computeRunningCost(float* s, float* u, float* du, float* vars, int timestep);
  __device__ float terminalCost(float* s);

  CLASS_T* cost_d_ = nullptr;
protected:
  PARAMS_T params_;

};

#endif // COSTS_CUH_
