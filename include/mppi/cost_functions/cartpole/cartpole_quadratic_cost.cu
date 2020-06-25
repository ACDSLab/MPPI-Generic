#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>

CartpoleQuadraticCost::CartpoleQuadraticCost(cudaStream_t stream) {
  bindToStream(stream);
}

float CartpoleQuadraticCost::computeStateCost(const Eigen::Ref<const state_array> s) {
  return (s[0]-params_.desired_terminal_state[0])*(s[0]-params_.desired_terminal_state[0])*params_.cart_position_coeff +
         (s[1]-params_.desired_terminal_state[1])*(s[1]-params_.desired_terminal_state[1])*params_.cart_velocity_coeff +
         (s[2]-params_.desired_terminal_state[2])*(s[2]-params_.desired_terminal_state[2])*params_.pole_angle_coeff +
         (s[3]-params_.desired_terminal_state[3])*(s[3]-params_.desired_terminal_state[3])*params_.pole_angular_velocity_coeff;
}

__host__ __device__ float CartpoleQuadraticCost::getStateCost(float *state) {
  return (state[0]-params_.desired_terminal_state[0])*(state[0]-params_.desired_terminal_state[0])*params_.cart_position_coeff +
         (state[1]-params_.desired_terminal_state[1])*(state[1]-params_.desired_terminal_state[1])*params_.cart_velocity_coeff +
         (state[2]-params_.desired_terminal_state[2])*(state[2]-params_.desired_terminal_state[2])*params_.pole_angle_coeff +
         (state[3]-params_.desired_terminal_state[3])*(state[3]-params_.desired_terminal_state[3])*params_.pole_angular_velocity_coeff;
}

__host__ __device__ float CartpoleQuadraticCost::getControlCost(float *u, float *du, float *vars) {
  return params_.control_force_coeff*du[0]*(u[0] - du[0])/(vars[0]*vars[0]);
}

__host__ __device__ float CartpoleQuadraticCost::computeRunningCost(float *s, float *u, float *du, float *vars, int timestep) {
  return getStateCost(s) + getControlCost(u, du, vars);
}

__host__ __device__ float CartpoleQuadraticCost::terminalCost(float *state) {
  return ((state[0]-params_.desired_terminal_state[0])*(state[0]-params_.desired_terminal_state[0])*params_.cart_position_coeff +
         (state[1]-params_.desired_terminal_state[1])*(state[1]-params_.desired_terminal_state[1])*params_.cart_velocity_coeff +
         (state[2]-params_.desired_terminal_state[2])*(state[2]-params_.desired_terminal_state[2])*params_.pole_angle_coeff +
         (state[3]-params_.desired_terminal_state[3])*(state[3]-params_.desired_terminal_state[3])*params_.pole_angular_velocity_coeff)*params_.terminal_cost_coeff;
}