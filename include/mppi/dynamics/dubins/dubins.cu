#include <mppi/dynamics/dubins/dubins.cuh>

DubinsDynamics::DubinsDynamics(cudaStream_t stream) : Dynamics<DubinsDynamics, DubinsParams, 3, 2>(stream)
{
  this->params_ = DubinsParams();
}

void DubinsDynamics::computeDynamics(const Eigen::Ref<const state_array>& state,
                                     const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der)
{
  state_der(0) = control(0) * cos(state(2));
  state_der(1) = control(0) * sin(state(2));
  state_der(2) = control(1);
}

bool DubinsDynamics::computeGrad(const Eigen::Ref<const state_array>& state,
                                 const Eigen::Ref<const control_array>& control, Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B)
{
  A(0, 2) = -control(0) * sin(state(2));
  A(1, 2) = control(0) * cos(state(2));

  B(0, 0) = cos(state(2));
  B(1, 0) = sin(state(2));
  B(2, 1) = 1;
  return true;
}

void DubinsDynamics::updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt)
{
  state += state_der * dt;
  state(2) = angle_utils::normalizeAngle(state(2));
  state_der.setZero();
}

DubinsDynamics::state_array DubinsDynamics::interpolateState(const Eigen::Ref<state_array> state_1,
                                                             const Eigen::Ref<state_array> state_2, const double alpha)
{
  state_array result = (1 - alpha) * state_1 + alpha * state_2;
  result(2) = angle_utils::interpolateEulerAngleLinear(state_1(2), state_2(2), alpha);
  return result;
}

__device__ void DubinsDynamics::updateState(float* state, float* state_der, const float dt)
{
  int i;
  int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < STATE_DIM; i += blockDim.y)
  {
    state[i] += state_der[i] * dt;
    if (tdy == 2)
    {
      state[i] = angle_utils::normalizeAngle(state[i]);
    }
    state_der[i] = 0;  // Important: reset the state derivative to zero.
  }
}

__device__ void DubinsDynamics::computeDynamics(float* state, float* control, float* state_der, float* theta_s)
{
  state_der[0] = control[0] * cos(state[2]);
  state_der[1] = control[0] * sin(state[2]);
  state_der[2] = control[1];
}
