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
  Eigen::Vector4f q;

  Eigen::Map<const Eigen::Vector3f> x_g(this->params_.x_goal);
  Eigen::Map<const Eigen::Vector3f> v_g(this->params_.v_goal);
  Eigen::Map<const Eigen::Vector4f> q_g(this->params_.q_goal);
  Eigen::Map<const Eigen::Vector3f> w_g(this->params_.w_goal);

  x = s.block<3, 1>(0, 0);
  v = s.block<3, 1>(3, 0);
  q = s.block<4, 1>(6, 0);
  w = s.block<3, 1>(10, 0);

  state_array test = this->params_.getDesiredState();

  Eigen::Vector3f x_cost = this->params_.x_coeff * (x - x_g).array().square();
  Eigen::Vector3f v_cost = this->params_.v_coeff * (v - v_g).array().square();
  Eigen::Vector4f q_cost = this->params_.q_coeff * (q - q_g).array().square();
  Eigen::Vector3f w_cost = this->params_.w_coeff * (w - w_g).array().square();


  return x_cost.sum() + v_cost.sum() + q_cost.sum() + w_cost.sum();
}

float QuadrotorQuadraticCost::terminalCost(const Eigen::Ref<const state_array> s) {
  return this->params_.terminal_cost_coeff * computeStateCost(s);
}

/**
 * Devic Functions
 */
__device__ float QuadrotorQuadraticCost::computeStateCost(float* s) {
  float s_diff[STATE_DIM];
  int i;

  for(i = 0; i < STATE_DIM; i++) {
    s_diff[i] = powf(s[i] - this->params_.s_goal[i], 2);
  }

  for(i = 0; i < 3; i++) {
    s_diff[i] *= this->params_.x_coeff;
  }

  for(i = 3; i < 6; i++) {
    s_diff[i] *= this->params_.v_coeff;
  }

  for(i = 6; i < 10; i++) {
    s_diff[i] *= this->params_.q_coeff;
  }

  for(i = 10; i < 13; i++) {
    s_diff[i] *= this->params_.w_coeff;
  }

  float sum = 0;
  for(i = 0; i < STATE_DIM; i++) {
    sum += s_diff[i];
  }


  return sum;
}

__device__ float QuadrotorQuadraticCost::computeRunningCost(float* s, float* u,
                                                            float* du, float* variance,
                                                            int timestep) {
  return computeStateCost(s) + computeLikelihoodRatioCost(u, du, variance);
}

__device__ float QuadrotorQuadraticCost::terminalCost(float* s) {
  return this->params_.terminal_cost_coeff * computeStateCost(s);
}