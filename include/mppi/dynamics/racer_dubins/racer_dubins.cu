#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>

template <class CLASS_T, int STATE_DIM>
void RacerDubinsImpl<CLASS_T, STATE_DIM>::computeDynamics(const Eigen::Ref<const state_array>& state,
                                                          const Eigen::Ref<const control_array>& control,
                                                          Eigen::Ref<state_array> state_der)
{
  bool enable_brake = control(0) < 0;
  // applying position throttle
  state_der(0) = (!enable_brake) * this->params_.c_t[0] * control(0) +
                 (enable_brake) * this->params_.c_b[0] * control(0) * (state(0) >= 0 ? 1 : -1) -
                 this->params_.c_v[0] * state(0) + this->params_.c_0;
  state_der(1) = (state(0) / this->params_.wheel_base) * tan(state(4));
  state_der(2) = state(0) * cosf(state(1));
  state_der(3) = state(0) * sinf(state(1));
  state_der(4) = (control(1) * this->params_.steer_command_angle_scale - state(4)) * this->params_.steering_constant;
  state_der(4) = max(min(state_der(4), this->params_.max_steer_rate), -this->params_.max_steer_rate);
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
  state(4) = max(min(state(4), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  state(5) = state_der(4);
  state(6) = state_der(0);  // include accel in state
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
  for (i = tdy; i < 5; i += blockDim.y)
  {
    state[i] += state_der[i] * dt;
    if (i == 0)
    {
      state[6] = state_der[i];  // include accel in state
    }
    if (i == 1)
    {
      state[i] = angle_utils::normalizeAngle(state[i]);
    }
    if (i == 4)
    {
      state[i] = max(min(state[i], this->params_.max_steer_angle), -this->params_.max_steer_angle);
      state[5] = state_der[i];
    }
    state_der[i] = 0;  // Important: reset the state derivative to zero.
  }
}

template <class CLASS_T, int STATE_DIM>
Eigen::Quaternionf RacerDubinsImpl<CLASS_T, STATE_DIM>::get_attitude(const Eigen::Ref<const state_array>& state)
{
  float theta = state[1];
  return Eigen::Quaternionf(cos(theta / 2), 0, 0, sin(theta / 2));
}

template <class CLASS_T, int STATE_DIM>
Eigen::Vector3f RacerDubinsImpl<CLASS_T, STATE_DIM>::get_position(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[2], state[3], 0);
}

template <class CLASS_T, int STATE_DIM>
__device__ void RacerDubinsImpl<CLASS_T, STATE_DIM>::computeDynamics(float* state, float* control, float* state_der,
                                                                     float* theta_s)
{
  bool enable_brake = control[0] < 0;
  // applying position throttle
  state_der[0] = (!enable_brake) * this->params_.c_t[0] * control[0] +
                 (enable_brake) * this->params_.c_b[0] * control[0] * (state[0] >= 0 ? 1 : -1) -
                 this->params_.c_v[0] * state[0] + this->params_.c_0;
  state_der[1] = (state[0] / this->params_.wheel_base) * tan(state[4] / this->params_.steer_angle_scale[0]);
  state_der[2] = state[0] * cosf(state[1]);
  state_der[3] = state[0] * sinf(state[1]);
  state_der[4] = (control[1] * this->params_.steer_command_angle_scale - state[4]) * this->params_.steering_constant;
  state_der[4] = max(min(state_der[4], this->params_.max_steer_rate), -this->params_.max_steer_rate);
}

template <class CLASS_T, int STATE_DIM>
void RacerDubinsImpl<CLASS_T, STATE_DIM>::getStoppingControl(const Eigen::Ref<const state_array>& state,
                                                             Eigen::Ref<control_array> u)
{
  u[0] = -1.0;  // full brake
  u[1] = 0.0;   // no steering
}

template <class CLASS_T, int STATE_DIM>
void RacerDubinsImpl<CLASS_T, STATE_DIM>::enforceLeash(const Eigen::Ref<const state_array>& state_true,
                                                       const Eigen::Ref<const state_array>& state_nominal,
                                                       const Eigen::Ref<const state_array>& leash_values,
                                                       Eigen::Ref<state_array> state_output)
{
  // update state_output for leash, need to handle x and y positions specially, convert to body frame and leash in body
  // frame. transform x and y into body frame
  float dx = state_nominal[S_INDEX(POS_X)] - state_true[S_INDEX(POS_X)];
  float dy = state_nominal[S_INDEX(POS_Y)] - state_true[S_INDEX(POS_Y)];
  float dx_body = dx * cos(state_true[S_INDEX(YAW)]) + dy * sin(state_true[S_INDEX(YAW)]);
  float dy_body = -dx * sin(state_true[S_INDEX(YAW)]) + dy * cos(state_true[S_INDEX(YAW)]);

  // determine leash in body frame
  float y_leash = leash_values[S_INDEX(POS_Y)];
  float x_leash = leash_values[S_INDEX(POS_X)];
  dx_body = fminf(fmaxf(dx_body, -x_leash), x_leash);
  dy_body = fminf(fmaxf(dy_body, -y_leash), y_leash);

  // transform back to map frame
  dx = dx_body * cos(state_true[S_INDEX(YAW)]) + -dy_body * sin(state_true[S_INDEX(YAW)]);
  dy = dx_body * sin(state_true[S_INDEX(YAW)]) + dy_body * cos(state_true[S_INDEX(YAW)]);

  // apply leash
  state_output[S_INDEX(POS_X)] += dx;
  state_output[S_INDEX(POS_Y)] += dy;

  // handle leash for rest of states
  float diff;
  for (int i = 0; i < STATE_DIM; i++)
  {
    // use body x and y for leash
    if (i == S_INDEX(POS_X) || i == S_INDEX(POS_Y))
    {
      // handle outside for loop
      continue;
    }
    else if (i == S_INDEX(YAW))
    {
      diff = fabsf(angle_utils::shortestAngularDistance(state_nominal[i], state_true[i]));
    }
    else
    {
      diff = fabsf(state_nominal[i] - state_true[i]);
    }

    if (leash_values[i] < diff)
    {
      float leash_dir = fminf(fmaxf(state_nominal[i] - state_true[i], -leash_values[i]), leash_values[i]);
      state_output[i] = state_true[i] + leash_dir;
    }
    else
    {
      state_output[i] = state_nominal[i];
    }
  }
}
