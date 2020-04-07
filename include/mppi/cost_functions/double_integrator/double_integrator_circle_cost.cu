#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>

DoubleIntegratorCircleCost::DoubleIntegratorCircleCost(cudaStream_t stream) {
  bindToStream(stream);
}

DoubleIntegratorCircleCost::~DoubleIntegratorCircleCost() {
  if (!GPUMemStatus_) {
    freeCudaMem();
    GPUMemStatus_ = false;
  }
}

void DoubleIntegratorCircleCost::paramsToDevice() {
  HANDLE_ERROR(cudaMemcpyAsync(&cost_d_->params_, &params_, sizeof(DoubleIntegratorCircleCostParams), cudaMemcpyHostToDevice, stream_));
}

__host__ __device__ float DoubleIntegratorCircleCost::getStateCost(float *s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);

  float cost = 0;
  if ((radial_position < params_.inner_path_radius2) || (radial_position > params_.outer_path_radius2)) {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost*(current_velocity - params_.velocity_desired)*(current_velocity - params_.velocity_desired);
  return cost;

}

__host__ __device__ float DoubleIntegratorCircleCost::getControlCost(float *u, float *du, float *vars) {
  return du[0]*(u[0] - du[0])/(vars[0]*vars[0]);
}

__host__ __device__ float DoubleIntegratorCircleCost::computeRunningCost(float *s, float *u, float *du, float *vars, int timestep) {
  return getStateCost(s);
}

__host__ __device__ float DoubleIntegratorCircleCost::terminalCost(float *state) {
  return 0;
}