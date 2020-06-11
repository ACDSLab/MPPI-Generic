#include <mppi/cost_functions/quadrotor/quadrotor_quadratic_cost.cuh>

QuadrotorQuadraticCost::QuadrotorQuadraticCost(cudaStream_t stream) {
  bindToStream(stream);
}

QuadrotorQuadraticCost::~QuadrotorQuadraticCost(){
  if (!GPUMemStatus_) {
    freeCudaMem();
    GPUMemStatus_ = false;
  }
}

/**
 * Host Functions
 */
// void paramsToDevice();

float QuadrotorQuadraticCost::computeStateCost(const Eigen::Ref<const state_array> s) {
  Eigen::Vector3f x, v, w;
  x = s.block<3, 1>(0, 0);
  v = s.block<3, 1>(3, 0);

  return 0;
}

float QuadrotorQuadraticCost::terminalCost(const Eigen::Ref<const state_array> s) {
  return 0;
}

/**
 * Devic Functions
 */
__device__ float QuadrotorQuadraticCost::computeStateCost(float* s) {
  return 0;
}

__device__ float QuadrotorQuadraticCost::computeRunningCost(float* s, float* u,
                                                            float* du, float* variance,
                                                            int timestep) {
  return 0;
}

__device__ float QuadrotorQuadraticCost::terminalCost(float* s) {
  return 0;
}