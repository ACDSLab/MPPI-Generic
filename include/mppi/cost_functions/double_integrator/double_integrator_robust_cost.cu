#include <mppi/cost_functions/double_integrator/double_integrator_robust_cost.cuh>


// TODO Move into math_utils.h
/**
 * Linear interpolation
 * Given two coordinates (x_min, y_min) and (x_max, y_max)
 * And the x location of a third (x), return the y location
 * along the line between the two points
 */
__host__ __device__ float linInterp(float x, float x_min, float x_max, float y_min, float y_max) {
  return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min;
}

/**
 * Calculates the normalized distance from the centerline
 * @param r - current radius
 * @param r_inner - the inside radius of a track
 * @param r_outer - the outside radius of a track
 * @return norm_dist - a normalized distance away from the centerline
 * norm_dist = 0 -> on the centerline
 * norm_dist = 1 -> on one of the track boundaries inner, or outer
 */
__host__ __device__ float normDistFromCenter(float r, float r_inner, float r_outer) {
  float r_center = (r_inner + r_outer) / 2;
  float r_width = (r_outer - r_inner);
  float dist_from_center = fabsf(r - r_center);
  float norm_dist = dist_from_center / (r_width * 0.5);
  return norm_dist;
}

DoubleIntegratorRobustCost::DoubleIntegratorRobustCost(cudaStream_t stream) {
  bindToStream(stream);
}

__device__ float DoubleIntegratorRobustCost::computeStateCost(float* s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);
  float current_angular_momentum = s[0]*s[3] - s[1]*s[2];

  float cost = 0;
  float normalized_dist_from_center = normDistFromCenter(sqrt(radial_position),
    sqrt(params_.inner_path_radius2), sqrt(params_.outer_path_radius2));
  float steep_percent_boundary = 0.75;
  float steep_cost = 0.1 * params_.crash_cost; // 10 percent of crash cost
  // Shallow cost region
  if (normalized_dist_from_center <= steep_percent_boundary) {
    cost += linInterp(normalized_dist_from_center, 0, steep_percent_boundary,
                      0, steep_cost);
  }
  // Steep cost region
  if (normalized_dist_from_center > steep_percent_boundary &&
      normalized_dist_from_center <= 1.0) {
    cost += linInterp(normalized_dist_from_center, steep_percent_boundary, 1,
                      steep_cost, params_.crash_cost);
  }
  // Crash Cost region
  if (normalized_dist_from_center > 1.0) {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost *
    powf(current_velocity - params_.velocity_desired, 2);
  cost += params_.velocity_cost *
    powf(current_angular_momentum - params_.angular_momentum_desired, 2);
  return cost;
}

float DoubleIntegratorRobustCost::computeStateCost(const Eigen::Ref<const state_array> s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);
  float current_angular_momentum = s[0]*s[3] - s[1]*s[2];

  float cost = 0;
  float normalized_dist_from_center = normDistFromCenter(sqrt(radial_position),
    sqrt(params_.inner_path_radius2), sqrt(params_.outer_path_radius2));
  float steep_percent_boundary = 0.75;
  float steep_cost = 0.1 * params_.crash_cost;
  if (normalized_dist_from_center <= steep_percent_boundary) {
    cost += linInterp(normalized_dist_from_center, 0, steep_percent_boundary,
                      0, steep_cost);
  } else if (normalized_dist_from_center > steep_percent_boundary &&
             normalized_dist_from_center <= 1.0) {
    cost += linInterp(normalized_dist_from_center, steep_percent_boundary, 1,
                      steep_cost, params_.crash_cost);
  } else {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost *
    powf(current_velocity - params_.velocity_desired, 2);
  cost += params_.velocity_cost *
    powf(current_angular_momentum - params_.angular_momentum_desired, 2);
  return cost;
}

float DoubleIntegratorRobustCost::terminalCost(const Eigen::Ref<const state_array> s) {
  return 0;
}

__device__ float DoubleIntegratorRobustCost::terminalCost(float* s) {
  return 0;
}

__device__ float DoubleIntegratorRobustCost::computeRunningCost(float* s, float* u,
    float* du, float* variance, int timestep) {
  float cost = 0;
  return computeStateCost(s) + computeLikelihoodRatioCost(u, du, variance);
}