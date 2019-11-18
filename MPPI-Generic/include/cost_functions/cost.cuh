#pragma once
/*
Header file for costs
*/

#ifndef COSTS_CUH_
#define COSTS_CUH_

#include<Eigen/Dense>

#include <utils/managed.cuh>

class Cost : Managed
{
public:

  Cost();
  ~Cost();

  void loadParams();
  void paramsToDevice();
  void freeCudaMem();

  __device__ float controlCost(float* u, float* du);
  __device__ float stateCost(float* s);
  __device__ float crashCost(float* s);
  __device__ float mapCost(float* s);
  __device__ float computeCost(float* s, float* u, float* du);
};

#endif // COSTS_CUH_
