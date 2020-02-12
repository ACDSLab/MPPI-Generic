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

template<class CLASS_T, class PARAMS_T>
class Cost : public Managed
{
public:
  // struct namespaced by the class
  /*
  typedef struct {
    // fill in data here
  } CostParams;
   */

  Cost() = default;
  ~Cost() = default;

  /**
   * Updates the device version of the parameter structure
   */
  void updateDevice();

  /**
   * Updates the cost parameters
   * @param params
   */
  __host__ __device__ void setParams(PARAMS_T params) {
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
    //std::cout << __PRETTY_FUNCTION__ << std::endl;
  };

  /**
   * allocates all the cuda memory needed for the object
   */
  void allocateCudaMem();

  /**
   * deallocates the allocated cuda memory for an object
   */
  void freeCudaMem();

  inline __host__ __device__ PARAMS_T getParams() const {return this->params_;}

  __host__ __device__ float controlCost(float* u, float* du);
  __host__ __device__ float computeRunningCost(float* s, float* u, float* du);
  __host__ __device__ float terminalCost(float* s);
  __host__ __device__ float computeCost(float* s, float* u, float* du);

  CLASS_T* cost_d_ = nullptr;
protected:
  PARAMS_T params_;

};

#endif // COSTS_CUH_
