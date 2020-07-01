#ifndef AR_NN_DYNAMICS_CUH_
#define AR_NN_DYNAMICS_CUH_

#include <mppi/cost_functions/autorally/ar_standard_cost.cuh>

struct ARRobustCostParams : public ARStandardCostParams {
  //Miscellaneous
  int num_timesteps = 100;
  float heading_coeff = 0.0;

  ARRobustCostParams() {
    control_cost_coeff[0] = 0.0; // steering_coeff
    control_cost_coeff[1] = 0.0; // throttle_coeff
    //values, if negative ignored
    desired_speed = -1;
    max_slip_ang = 1.5;

    //Cost term coefficients
    track_coeff = 33.0;
    slip_coeff = 0.0;
    speed_coeff = 20.0;
    //Constraint penalty thresholds
    crash_coeff = 125000;
    boundary_threshold = 0.75;
    //Miscellaneous
    track_slop = 0;
  }
};

template<class CLASS_T = void, class PARAMS_T = ARRobustCostParams>
class ARRobustCost : public ARStandardCost< ARRobustCost<CLASS_T, PARAMS_T>, PARAMS_T> {
public:

  ARRobustCost(cudaStream_t stream=0);// : ARStandardCost<PARAMS_T>(steam);
  ~ARRobustCost();

  __host__ __device__ float getStabilizingCost(float* s);
  __device__ float getCostmapCost(float* s);
  __device__ float computeStateCost(float *s);
  __device__ float computeRunningCost(float* s, float* u, float* du, float* vars, float lambda, float alpha, int timestep);




private:
};

#if __CUDACC__
#include "ar_robust_cost.cu"
#endif

#endif // AR_NN_DYNAMICS_CUH_
