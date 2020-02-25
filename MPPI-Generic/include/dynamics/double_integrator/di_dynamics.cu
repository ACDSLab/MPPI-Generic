#include <dynamics/double_integrator/di_dynamics.cuh>

DoubleIntegratorDynamics::DoubleIntegratorDynamics(float system_noise, cudaStream_t stream) :
Dynamics<DoubleIntegratorDynamics, DoubleIntegratorParams, 4, 2>(stream) {
  this->params_ = DoubleIntegratorParams(system_noise);
}

DoubleIntegratorDynamics::~DoubleIntegratorDynamics() = default;;

void DoubleIntegratorDynamics::computeDynamics(const Eigen::Ref<const state_array> &state,
        const Eigen::Ref<const control_array> &control, Eigen::Ref<state_array> state_der) {
  state_der(0) = state(2); // xdot;
  state_der(1) = state(3); // ydot;
  state_der(2) = control(0); // x_force;
  state_der(3) = control(1); // y_force
}

void DoubleIntegratorDynamics::computeGrad(const Eigen::Ref<const state_array> &state,
        const Eigen::Ref<const control_array> &control,
        Eigen::Ref<dfdx> A,
        Eigen::Ref<dfdu> B) {
  A(0,2) = 1;
  A(1,3) = 1;

  B(2,0) = 1;
  B(3,1) = 1;
}

void DoubleIntegratorDynamics::printState(float *state) {
  printf("X position: %f; Y position: %f; X velocity: %f; Y velocity: %f \n", state[0], state[1], state[2], state[3]);
}

__device__ void DoubleIntegratorDynamics::computeDynamics(float* state, float* control,
                                                  float* state_der, float* theta_s)
{
  state_der[0] = state[2]; // xdot;
  state_der[1] = state[3]; // ydot;
  state_der[2] = control[0]; // x_force;
  state_der[3] = control[1]; // y_force
}