#include <mppi/cost_functions/double_integrator/double_integrator_robust_cost.cuh>

__device__ float linInterp(float x, float x_min, float x_max, float y_min, float y_max) {
  return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min;
}

__device__ float normDistFromCenter(float r, float r_innter, float r_outer) {
  float r_center = (r_inner + r_outer) / 2;
  float r_width = (r_outer - r_inner);
  float dist_from_center = fabsf(r - r_center);
  float norm_dist = dist_from_center / (r_width * 0.5);
  return norm_dist;
}

__device__ DoubleIntegratorRobustCost::computeStateCost(float* s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);
  float current_angular_momentum = s[0]*s[3] - s[1]*s[2];

  float cost = 0;
  // float ratio_dist_to_inner = radial_position;
  // float center_radius = (sqrtf(params_.inner_path_radius2) + sqrtf(params_.outer_path_radius2)) / 2;
  // float track_width = (sqrtf(params_.outer_path_radius2) - sqrtf(params_.inner_path_radius2)) / 2;
  // float dist_from_center = fabsf(sqrtf(radial_position) - center_radius);
  // float normalized_dist_from_center = dist_from_center / (track_width * 0.5);
  float normalized_dist_from_center = normDistFromCenter(sqrt(radial_position),
    sqrt(params_.inner_path_radius2), sqrt(params_.outer_path_radius2));
  float steep_percent_boundary = 0.75;
  float steep_cost = 0.1 * params_.crash_cost; // 10 percent of crash cost
  // Shallow cost for
  if (normalized_dist_from_center <= steep_percent_boundary) {
    cost += linInterp(normalized_dist_from_center, 0, steep_percent_boundary,
                       0, steep_cost);
  }
  if (normalized_dist_from_center > steep_percent_boundary &&
      normalized_dist_from_center <= 1.0) {
    cost += linInterp(normalized_dist_from_center, steep_percent_boundary, 1,
                       steep_cost, params_.crash_cost);
  }
  if (normalized_dist_from_center > 1.0) {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost*(current_velocity - params_.velocity_desired)*(current_velocity - params_.velocity_desired);
  cost += params_.velocity_cost*(current_angular_momentum - params_.angular_momentum_desired)*(current_angular_momentum - params_.angular_momentum_desired);
  return cost;
}