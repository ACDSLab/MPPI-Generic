#ifndef AR_NN_DYNAMICS_CUH_
#define AR_NN_DYNAMICS_CUH_

#include <cost_functions/autorally/ar_standard_cost.cuh>

typedef struct {
  //values, if negative ignored
  float desired_speed = -1;
  //Cost term coefficients
  float speed_coeff = 20.0;
  float track_coeff = 33.0;
  float heading_coeff = 0.0;
  float steering_coeff = 0.0;
  float throttle_coeff = 0.0;
  float slip_coeff = 0.0;
  //Constraint penalty thresholds
  float crash_coeff = 125000;
  float boundary_threshold = 0.75;
  float max_slip_ang = 1.5;
  //Miscellaneous
  float track_slop = 0.0;
  int num_timesteps = 100;
  float3 r_c1;
  float3 r_c2;
  float3 trs;
} ARRobustCostParams;

template<class CLASS_T = void, class PARAMS_T = ARRobustCostParams>
class ARRobustCost : public ARStandardCost< ARRobustCost<CLASS_T, PARAMS_T>, PARAMS_T> {
public:

  ARRobustCost(cudaStream_t stream=0);// : ARStandardCost<PARAMS_T>(steam);
  ~ARRobustCost();

  __host__ __device__ float getStabilizingCost(float* s);
  __device__ float getCostmapCost(float* s);
  __device__ float computeCost(float* s, float* u, float* du, float* vars, int* crash, int timestep);



private:
};

#if __CUDACC__
#include "ar_robust_cost.cu"
#endif

#endif // AR_NN_DYNAMICS_CUH_
