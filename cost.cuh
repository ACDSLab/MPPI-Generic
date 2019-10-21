/*
Header file for costs
*/

#include<Eigen/Dense>

class Cost
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
}
