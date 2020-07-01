#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>

DoubleIntegratorCircleCost::DoubleIntegratorCircleCost(cudaStream_t stream) {
  bindToStream(stream);
}

__device__ float DoubleIntegratorCircleCost::computeStateCost(float *s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);
  float current_angular_momentum = s[0]*s[3] - s[1]*s[2];

  float cost = 0;
  if ((radial_position < params_.inner_path_radius2) ||
      (radial_position > params_.outer_path_radius2)) {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost * powf(current_velocity - params_.velocity_desired, 2) *
  cost += params_.velocity_cost * powf(current_angular_momentum - params_.angular_momentum_desired, 2)
  return cost;
}

float DoubleIntegratorCircleCost::computeStateCost(const Eigen::Ref<const state_array> s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);
  float current_angular_momentum = s[0]*s[3] - s[1]*s[2];

  float cost = 0;
  if ((radial_position < params_.inner_path_radius2) ||
      (radial_position > params_.outer_path_radius2)) {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost * powf(current_velocity - params_.velocity_desired, 2) *
  cost += params_.velocity_cost * powf(current_angular_momentum - params_.angular_momentum_desired, 2)
  return cost;
}

float DoubleIntegratorCircleCost::terminalCost(const Eigen::Ref<const state_array> s) {
  return 0;
}

__device__ float DoubleIntegratorCircleCost::terminalCost(float *state) {
  return 0;
}
