#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>

template <class CLASS_T, int STATE_DIM>
void RacerDubinsImpl<CLASS_T, STATE_DIM>::computeDynamics(const Eigen::Ref<const state_array>& state,
                                                          const Eigen::Ref<const control_array>& control,
                                                          Eigen::Ref<state_array> state_der)
{
  bool enable_brake = control(0) < 0;
  // applying position throttle
  state_der(0) = (!enable_brake) * this->params_.c_t * control(0) +
                 (enable_brake) * this->params_.c_b * control(0) * state(0) - this->params_.c_v * state(0) +
                 this->params_.c_0;
  state_der(1) = (state(0) / this->params_.wheel_base) * tan(state(4));
  state_der(2) = state(0) * cosf(state(1));
  state_der(3) = state(0) * sinf(state(1));
  state_der(4) = control(1) / this->params_.steer_command_angle_scale;
}

template <class CLASS_T, int STATE_DIM>
bool RacerDubinsImpl<CLASS_T, STATE_DIM>::computeGrad(const Eigen::Ref<const state_array>& state,
                                                      const Eigen::Ref<const control_array>& control,
                                                      Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B)
{
  return false;
}

template <class CLASS_T, int STATE_DIM>
void RacerDubinsImpl<CLASS_T, STATE_DIM>::updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der,
                                                      const float dt)
{
  state += state_der * dt;
  state(1) = angle_utils::normalizeAngle(state(1));
  state(4) -= state_der(4) * dt;
  state(4) = state_der(4) + (state(4) - state_der(4)) * expf(-this->params_.steering_constant * dt);
  state_der.setZero();
}

template <class CLASS_T, int STATE_DIM>
RacerDubinsImpl<CLASS_T, STATE_DIM>::state_array RacerDubinsImpl<CLASS_T, STATE_DIM>::interpolateState(
    const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2, const float alpha)
{
  state_array result = (1 - alpha) * state_1 + alpha * state_2;
  result(1) = angle_utils::interpolateEulerAngleLinear(state_1(1), state_2(1), alpha);
  return result;
}

template <class CLASS_T, int STATE_DIM>
__device__ void RacerDubinsImpl<CLASS_T, STATE_DIM>::updateState(float* state, float* state_der, const float dt)
{
  int i;
  int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < STATE_DIM; i += blockDim.y)
  {
    state[i] += state_der[i] * dt;
    if (i == 1)
    {
      state[i] = angle_utils::normalizeAngle(state[i]);
    }
    if (i == 4)
    {
      state[i] -= state_der[i] * dt;
      state[i] = state_der[i] + (state[i] - state_der[i]) * expf(-this->params_.steering_constant * dt);
      // state[i] += state_der[i] * expf(-this->params_.steering_constant * dt);
    }
    state_der[i] = 0;  // Important: reset the state derivative to zero.
  }
}

template <class CLASS_T, int STATE_DIM>
__device__ void RacerDubinsImpl<CLASS_T, STATE_DIM>::computeDynamics(float* state, float* control, float* state_der,
                                                                     float* theta_s)
{
  bool enable_brake = control[0] < 0;
  // applying position throttle
  state_der[0] = (!enable_brake) * this->params_.c_t * control[0] +
                 (enable_brake) * this->params_.c_b * control[0] * state[0] - this->params_.c_v * state[0] +
                 this->params_.c_0;
  state_der[1] = (state[0] / this->params_.wheel_base) * tan(state[4]);
  state_der[2] = state[0] * cosf(state[1]);
  state_der[3] = state[0] * sinf(state[1]);
  state_der[4] = control[1] / this->params_.steer_command_angle_scale;
}
