#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/utils/math_utils.h>

template <class CLASS_T, class PARAMS_T>
void RacerDubinsImpl<CLASS_T, PARAMS_T>::computeDynamics(const Eigen::Ref<const state_array>& state,
                                                         const Eigen::Ref<const control_array>& control,
                                                         Eigen::Ref<state_array> state_der)
{
  bool enable_brake = control(C_INDEX(THROTTLE_BRAKE)) < 0;

  state_der(S_INDEX(BRAKE_STATE)) =
      min(max((enable_brake * -control(C_INDEX(THROTTLE_BRAKE)) - state(S_INDEX(BRAKE_STATE))) *
                  this->params_.brake_delay_constant,
              -this->params_.max_brake_rate_neg),
          this->params_.max_brake_rate_pos);
  // applying position throttle
  state_der(S_INDEX(VEL_X)) =
      (!enable_brake) * this->params_.c_t[0] * control(C_INDEX(THROTTLE_BRAKE)) * this->params_.gear_sign +
      this->params_.c_b[0] * state(S_INDEX(BRAKE_STATE)) * (state(S_INDEX(VEL_X)) >= 0 ? -1 : 1) -
      this->params_.c_v[0] * state(S_INDEX(VEL_X)) + this->params_.c_0;
  state_der(S_INDEX(YAW)) = (state(S_INDEX(VEL_X)) / this->params_.wheel_base) *
                            tan(state(S_INDEX(STEER_ANGLE)) / this->params_.steer_angle_scale);
  state_der(S_INDEX(POS_X)) = state(S_INDEX(VEL_X)) * cosf(state(S_INDEX(YAW)));
  state_der(S_INDEX(POS_Y)) = state(S_INDEX(VEL_X)) * sinf(state(S_INDEX(YAW)));
  state_der(S_INDEX(STEER_ANGLE)) =
      (control(C_INDEX(STEER_CMD)) * this->params_.steer_command_angle_scale - state(S_INDEX(STEER_ANGLE))) *
      this->params_.steering_constant;
  state_der(S_INDEX(STEER_ANGLE)) =
      max(min(state_der(S_INDEX(STEER_ANGLE)), this->params_.max_steer_rate), -this->params_.max_steer_rate);
}

template <class CLASS_T, class PARAMS_T>
bool RacerDubinsImpl<CLASS_T, PARAMS_T>::computeGrad(const Eigen::Ref<const state_array>& state,
                                                     const Eigen::Ref<const control_array>& control, Eigen::Ref<dfdx> A,
                                                     Eigen::Ref<dfdu> B)
{
  return false;
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsImpl<CLASS_T, PARAMS_T>::updateState(const Eigen::Ref<const state_array> state,
                                                     Eigen::Ref<state_array> next_state,
                                                     Eigen::Ref<state_array> state_der, const float dt)
{
  // Segmented it to ensure that roll and pitch don't get overwritten
  for (int i = 0; i < 6; i++)
  {
    next_state[i] = state[i] + state_der[i] * dt;
  }
  next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
  next_state(S_INDEX(STEER_ANGLE)) =
      max(min(next_state(S_INDEX(STEER_ANGLE)), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  next_state(S_INDEX(STEER_ANGLE_RATE)) = state_der(S_INDEX(STEER_ANGLE));
  next_state(S_INDEX(BRAKE_STATE)) =
      min(max(next_state(S_INDEX(BRAKE_STATE)), 0.0f), -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x);
}

template <class CLASS_T, class PARAMS_T>
RacerDubinsImpl<CLASS_T, PARAMS_T>::state_array RacerDubinsImpl<CLASS_T, PARAMS_T>::interpolateState(
    const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2, const float alpha)
{
  state_array result = (1 - alpha) * state_1 + alpha * state_2;
  result(S_INDEX(YAW)) = angle_utils::interpolateEulerAngleLinear(state_1(S_INDEX(YAW)), state_2(S_INDEX(YAW)), alpha);
  return result;
}

template <class CLASS_T, class PARAMS_T>
__device__ void RacerDubinsImpl<CLASS_T, PARAMS_T>::updateState(float* state, float* next_state, float* state_der,
                                                                const float dt)
{
  int i;
  int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < 6; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    if (i == S_INDEX(YAW))
    {
      next_state[i] = angle_utils::normalizeAngle(next_state[i]);
    }
    if (i == S_INDEX(STEER_ANGLE))
    {
      next_state[i] = max(min(next_state[i], this->params_.max_steer_angle), -this->params_.max_steer_angle);
      next_state[S_INDEX(STEER_ANGLE_RATE)] = state_der[i];
    }
    if (i == S_INDEX(BRAKE_STATE))
    {
      next_state[i] = min(max(next_state[i], 0.0f), 1.0f);
    }
  }
}

template <class CLASS_T, class PARAMS_T>
Eigen::Quaternionf RacerDubinsImpl<CLASS_T, PARAMS_T>::attitudeFromState(const Eigen::Ref<const state_array>& state)
{
  float theta = state[S_INDEX(YAW)];
  return Eigen::Quaternionf(cos(theta / 2), 0, 0, sin(theta / 2));
}

template <class CLASS_T, class PARAMS_T>
Eigen::Vector3f RacerDubinsImpl<CLASS_T, PARAMS_T>::positionFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(POS_X)], state[S_INDEX(POS_Y)], 0);
}

template <class CLASS_T, class PARAMS_T>
Eigen::Vector3f RacerDubinsImpl<CLASS_T, PARAMS_T>::velocityFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(VEL_X)], 0, 0);
}

template <class CLASS_T, class PARAMS_T>
Eigen::Vector3f RacerDubinsImpl<CLASS_T, PARAMS_T>::angularRateFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(0, 0, 0);  // TODO compute yaw rate from steering angle & vel
}

template <class CLASS_T, class PARAMS_T>
RacerDubinsImpl<CLASS_T, PARAMS_T>::state_array RacerDubinsImpl<CLASS_T, PARAMS_T>::stateFromOdometry(
    const Eigen::Quaternionf& q, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel, const Eigen::Vector3f& omega)
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

template <class CLASS_T, class PARAMS_T>
__device__ void RacerDubinsImpl<CLASS_T, PARAMS_T>::computeDynamics(float* state, float* control, float* state_der,
                                                                    float* theta_s)
{
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;

  state_der[S_INDEX(BRAKE_STATE)] =
      min(max((enable_brake * -control[C_INDEX(THROTTLE_BRAKE)] - state[S_INDEX(BRAKE_STATE)]) *
                  this->params_.brake_delay_constant,
              -this->params_.max_brake_rate_neg),
          this->params_.max_brake_rate_pos);

  // applying position throttle
  state_der[S_INDEX(VEL_X)] =
      (!enable_brake) * this->params_.c_t[0] * control[0] * this->params_.gear_sign +
      this->params_.c_b[0] * state[S_INDEX(BRAKE_STATE)] * (state[S_INDEX(VEL_X)] >= 0 ? -1 : 1) -
      this->params_.c_v[0] * state[S_INDEX(VEL_X)] + this->params_.c_0;
  state_der[S_INDEX(YAW)] = (state[S_INDEX(VEL_X)] / this->params_.wheel_base) *
                            tan(state[S_INDEX(STEER_ANGLE)] / this->params_.steer_angle_scale);
  state_der[S_INDEX(POS_X)] = state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] = state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min((control[1] * this->params_.steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
                  this->params_.steering_constant,
              this->params_.max_steer_rate),
          -this->params_.max_steer_rate);
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsImpl<CLASS_T, PARAMS_T>::getStoppingControl(const Eigen::Ref<const state_array>& state,
                                                            Eigen::Ref<control_array> u)
{
  u[0] = -1.0;  // full brake
  u[1] = 0.0;   // no steering
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsImpl<CLASS_T, PARAMS_T>::enforceLeash(const Eigen::Ref<const state_array>& state_true,
                                                      const Eigen::Ref<const state_array>& state_nominal,
                                                      const Eigen::Ref<const state_array>& leash_values,
                                                      Eigen::Ref<state_array> state_output)
{
  state_output = state_true;

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
  for (int i = 0; i < PARENT_CLASS::STATE_DIM; i++)
  {
    // use body x and y for leash
    if (i == S_INDEX(POS_X) || i == S_INDEX(POS_Y))
    {
      // handle outside for loop
      continue;
    }
    else if (i == S_INDEX(YAW))
    {
      diff = angle_utils::shortestAngularDistance(state_true[i], state_nominal[i]);
    }
    else
    {
      diff = state_nominal[i] - state_true[i];
    }

    if (leash_values[i] < fabsf(diff))
    {
      float leash_dir = fminf(fmaxf(diff, -leash_values[i]), leash_values[i]);
      state_output[i] = state_true[i] + leash_dir;
      if (i == S_INDEX(YAW))
      {
        state_output[i] = angle_utils::normalizeAngle(state_output[i]);
      }
    }
    else
    {
      state_output[i] = state_nominal[i];
    }
  }
}

template <class CLASS_T, class PARAMS_T>
RacerDubinsImpl<CLASS_T, PARAMS_T>::state_array
RacerDubinsImpl<CLASS_T, PARAMS_T>::stateFromMap(const std::map<std::string, float>& map)
{
  state_array s = state_array::Zero();
  if (map.find("VEL_X") == map.end() || map.find("VEL_Y") == map.end() || map.find("POS_X") == map.end() ||
      map.find("POS_Y") == map.end())
  {
    return s;
  }
  s(S_INDEX(POS_X)) = map.at("POS_X");
  s(S_INDEX(POS_Y)) = map.at("POS_Y");
  s(S_INDEX(VEL_X)) = map.at("VEL_X");
  s(S_INDEX(YAW)) = map.at("YAW");
  if (map.find("STEER_ANGLE") != map.end())
  {
    s(S_INDEX(STEER_ANGLE)) = map.at("STEER_ANGLE");
    s(S_INDEX(STEER_ANGLE_RATE)) = map.at("STEER_ANGLE_RATE");
  }
  else
  {
    s(S_INDEX(STEER_ANGLE)) = 0;
    s(S_INDEX(STEER_ANGLE_RATE)) = 0;
  }
  if (map.find("BRAKE_STATE") != map.end())
  {
    s(S_INDEX(BRAKE_STATE)) = map.at("BRAKE_STATE");
  }
  else if (map.find("BRAKE_CMD") != map.end())
  {
    s(S_INDEX(BRAKE_STATE)) = map.at("BRAKE_CMD");
  }
  else
  {
    s(S_INDEX(BRAKE_STATE)) = 0;
  }

  return s;
}

template <class OUTPUT_T, class TEX_T>
__device__ __host__ void RACER::computeStaticSettling(TEX_T* tex_helper, const float yaw, const float x, const float y,
                                                      float& roll, float& pitch, float* output)
{
  float height = 0.0f;

  float3 front_left = make_float3(2.981f, 0.737f, 0.0f);
  float3 front_right = make_float3(2.981f, -0.737f, 0.f);
  float3 rear_left = make_float3(0.0f, 0.737f, 0.0f);
  float3 rear_right = make_float3(0.0f, -0.737f, 0.0f);
  float3 body_pose = make_float3(x, y, 0.0f);
  float3 rotation = make_float3(roll, pitch, yaw);
  // front_left = make_float3(front_left.x * cosf(yaw) - front_left.y * sinf(yaw) + x,
  //                          front_left.x * sinf(yaw) + front_left.y * cosf(yaw) + y, 0.0f);
  // front_right = make_float3(front_right.x * cosf(yaw) - front_right.y * sinf(yaw) + x,
  //                           front_right.x * sinf(yaw) + front_right.y * cosf(yaw) + y, 0.0f);
  // rear_left = make_float3(rear_left.x * cosf(yaw) - rear_left.y * sinf(yaw) + x,
  //                         rear_left.x * sinf(yaw) + rear_left.y * cosf(yaw) + y, 0.0f);
  // rear_right = make_float3(rear_right.x * cosf(yaw) - rear_right.y * sinf(yaw) + x,
  //                          rear_right.x * sinf(yaw) + rear_right.y * cosf(yaw) + y, 0.0f);
  float front_left_height = 0.0f;
  float front_right_height = 0.0f;
  float rear_left_height = 0.0f;
  float rear_right_height = 0.0f;

  if (tex_helper->checkTextureUse(0))
  {
    front_left_height = tex_helper->queryTextureAtWorldOffsetPose(0, body_pose, front_left, rotation);
    front_right_height = tex_helper->queryTextureAtWorldOffsetPose(0, body_pose, front_right, rotation);
    rear_left_height = tex_helper->queryTextureAtWorldOffsetPose(0, body_pose, rear_left, rotation);
    rear_right_height = tex_helper->queryTextureAtWorldOffsetPose(0, body_pose, rear_right, rotation);

    float front_diff = front_left_height - front_right_height;
    front_diff = max(min(front_diff, 0.736f * 2.0f), -0.736f * 2.0f);
    float rear_diff = rear_left_height - rear_right_height;
    rear_diff = max(min(rear_diff, 0.736f * 2.0f), -0.736f * 2.0f);
    float front_roll = asinf(front_diff / (0.737f * 2.0f));
    float rear_roll = asinf(rear_diff / (0.737f * 2.0f));
    roll = (front_roll + rear_roll) / 2.0f;

    float left_diff = rear_left_height - front_left_height;
    left_diff = max(min(left_diff, 2.98f), -2.98f);
    float right_diff = rear_right_height - front_right_height;
    right_diff = max(min(right_diff, 2.98f), -2.98f);
    float left_pitch = asinf((left_diff) / 2.981f);
    float right_pitch = asinf((right_diff) / 2.981f);
    pitch = (left_pitch + right_pitch) / 2.0f;

    height = (rear_left_height + rear_right_height) / 2.0f;
  }
  else
  {
    roll = 0.0f;
    pitch = 0.0f;
    height = 0.0f;
  }

  // using 2pi so any rotation that accidently uses this will be using identity
  if (isnan(roll) || isinf(roll) || fabsf(roll) > M_PIf32)
  {
    roll = 2.0f * M_PIf32;
  }
  if (isnan(pitch) || isinf(pitch) || fabsf(pitch) > M_PIf32)
  {
    pitch = 2.0f * M_PIf32;
  }
  if (isnan(height) || isinf(height))
  {
    height = 0.0f;
  }

  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_FL_X)] = front_left.x;
  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_FL_Y)] = front_left.y;
  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_FR_X)] = front_right.x;
  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_FR_Y)] = front_right.y;
  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_RL_X)] = rear_left.x;
  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_RL_Y)] = rear_left.y;
  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_RR_X)] = rear_right.x;
  output[E_INDEX(OUTPUT_T, WHEEL_POS_I_RR_Y)] = rear_right.y;
  output[E_INDEX(OUTPUT_T, BASELINK_POS_I_Z)] = height;
}
