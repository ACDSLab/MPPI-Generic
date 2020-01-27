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


class Cost : public Managed
{
public:
  // struct namespaced by the class
  typedef struct {
    // fill in data here
  } CostParams;

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
  void updateParams(); // TODO what to pass in for the default

  /**
   * allocates all the cuda memory needed for the object
   */
  void allocateCudaMem();

  /**
   * deallocates the allocated cuda memory for an object
   */
  void freeCudaMem();

  __host__ __device__ float controlCost(float* u, float* du);
  __host__ __device__ float computeRunningCost(float* s, float* u, float* du);
  __host__ __device__ float terminalCost(float* s);
  __host__ __device__ float computeCost(float* s, float* u, float* du);
};

#endif // COSTS_CUH_
