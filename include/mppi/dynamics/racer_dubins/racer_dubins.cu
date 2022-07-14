#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/utils/math_utils.h>

template <class CLASS_T>
void RacerDubinsImpl<CLASS_T>::computeDynamics(const Eigen::Ref<const state_array>& state,
                                               const Eigen::Ref<const control_array>& control,
                                               Eigen::Ref<state_array> state_der)
{
  bool enable_brake = control(C_INDEX(THROTTLE_BRAKE)) < 0;
  // applying position throttle
  state_der(S_INDEX(VEL_X)) =
      (!enable_brake) * this->params_.c_t * control(C_INDEX(THROTTLE_BRAKE)) +
      (enable_brake) * this->params_.c_b * control(C_INDEX(THROTTLE_BRAKE)) * (state(S_INDEX(VEL_X)) >= 0 ? 1 : -1) -
      this->params_.c_v * state(S_INDEX(VEL_X)) + this->params_.c_0;
  state_der(S_INDEX(YAW)) = (state(S_INDEX(VEL_X)) / this->params_.wheel_base) * tan(state(4));
  state_der(S_INDEX(POS_X)) = state(S_INDEX(VEL_X)) * cosf(state(S_INDEX(YAW)));
  state_der(S_INDEX(POS_Y)) = state(S_INDEX(VEL_X)) * sinf(state(S_INDEX(YAW)));
  state_der(S_INDEX(STEER_ANGLE)) = control(C_INDEX(STEER_CMD)) / this->params_.steer_command_angle_scale;
}

template <class CLASS_T>
bool RacerDubinsImpl<CLASS_T>::computeGrad(const Eigen::Ref<const state_array>& state,
                                           const Eigen::Ref<const control_array>& control, Eigen::Ref<dfdx> A,
                                           Eigen::Ref<dfdu> B)
{
  return false;
}

template <class CLASS_T>
void RacerDubinsImpl<CLASS_T>::updateState(const Eigen::Ref<const state_array> state,
                                           Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
                                           const float dt)
{
  next_state = state + state_der * dt;
  next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
  next_state(S_INDEX(STEER_ANGLE)) -= state_der(S_INDEX(STEER_ANGLE)) * dt;
  next_state(S_INDEX(STEER_ANGLE)) =
      state_der(S_INDEX(STEER_ANGLE)) + (next_state(S_INDEX(STEER_ANGLE)) - state_der(S_INDEX(STEER_ANGLE))) *
                                            expf(-this->params_.steering_constant * dt);
  state_der.setZero();
}

template <class CLASS_T>
RacerDubinsImpl<CLASS_T>::state_array RacerDubinsImpl<CLASS_T>::interpolateState(const Eigen::Ref<state_array> state_1,
                                                                                 const Eigen::Ref<state_array> state_2,
                                                                                 const float alpha)
{
  state_array result = (1 - alpha) * state_1 + alpha * state_2;
  result(1) = angle_utils::interpolateEulerAngleLinear(state_1(1), state_2(1), alpha);
  return result;
}

template <class CLASS_T>
__device__ void RacerDubinsImpl<CLASS_T>::updateState(float* state, float* next_state, float* state_der, const float dt)
{
  int i;
  int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < PARENT_CLASS::STATE_DIM; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    if (i == S_INDEX(YAW))
    {
      next_state[i] = angle_utils::normalizeAngle(next_state[i]);
    }
    if (i == S_INDEX(STEER_ANGLE))
    {
      next_state[i] -= state_der[i] * dt;
      next_state[i] = state_der[i] + (next_state[i] - state_der[i]) * expf(-this->params_.steering_constant * dt);
      // next_state[i] += state_der[i] * expf(-this->params_.steering_constant * dt);
    }
    state_der[i] = 0;  // Important: reset the next_state derivative to zero.
  }
}

template <class CLASS_T>
Eigen::Quaternionf RacerDubinsImpl<CLASS_T>::attitudeFromState(const Eigen::Ref<const state_array>& state)
{
  float theta = state[S_INDEX(YAW)];
  return Eigen::Quaternionf(cos(theta / 2), 0, 0, sin(theta / 2));
}

template <class CLASS_T>
Eigen::Vector3f RacerDubinsImpl<CLASS_T>::positionFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(POS_X)], state[S_INDEX(POS_Y)], 0);
}

template <class CLASS_T>
Eigen::Vector3f RacerDubinsImpl<CLASS_T>::velocityFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(VEL_X)], 0, 0);
}

template <class CLASS_T>
Eigen::Vector3f RacerDubinsImpl<CLASS_T>::angularRateFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(0, 0, 0);  // TODO compute yaw rate from steering angle & vel
}

template <class CLASS_T>
RacerDubinsImpl<CLASS_T>::state_array RacerDubinsImpl<CLASS_T>::stateFromOdometry(const Eigen::Quaternionf& q,
                                                                                  const Eigen::Vector3f& pos,
                                                                                  const Eigen::Vector3f& vel,
                                                                                  const Eigen::Vector3f& omega)
{
  state_array s;
  s.setZero();
  s[S_INDEX(POS_X)] = pos[0];
  s[S_INDEX(POS_Y)] = pos[1];
  s[S_INDEX(VEL_X)] = vel[0];
  float _roll, _pitch, yaw;
  mppi::math::Quat2EulerNWU(q, _roll, _pitch, yaw);
  s[S_INDEX(YAW)] = yaw;
  return s;
}

template <class CLASS_T>
__device__ void RacerDubinsImpl<CLASS_T>::computeDynamics(float* state, float* control, float* state_der,
                                                          float* theta_s)
{
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  // applying position throttle
  state_der[S_INDEX(VEL_X)] =
      (!enable_brake) * this->params_.c_t * control[C_INDEX(THROTTLE_BRAKE)] +
      (enable_brake) * this->params_.c_b * control[C_INDEX(THROTTLE_BRAKE)] * (state[S_INDEX(VEL_X)] >= 0 ? 1 : -1) -
      this->params_.c_v * state[S_INDEX(VEL_X)] + this->params_.c_0;
  state_der[S_INDEX(YAW)] = (state[S_INDEX(VEL_X)] / this->params_.wheel_base) * tan(state[4]);
  state_der[S_INDEX(POS_X)] = state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] = state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(STEER_ANGLE)] = control[C_INDEX(STEER_CMD)] / this->params_.steer_command_angle_scale;
}

template <class CLASS_T>
void RacerDubinsImpl<CLASS_T>::getStoppingControl(const Eigen::Ref<const state_array>& state,
                                                  Eigen::Ref<control_array> u)
{
  u[0] = -1.0;  // full brake
  u[1] = 0.0;   // no steering
}
