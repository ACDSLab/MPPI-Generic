#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>
#include <mppi/utils/math_utils.h>

template <class CLASS_T>
void RacerDubinsElevationImpl<CLASS_T>::GPUSetup()
{
  PARENT_CLASS::GPUSetup();
  tex_helper_->GPUSetup();
  // makes sure that the device ptr sees the correct texture object
  HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->tex_helper_), &(tex_helper_->ptr_d_),
                               sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
}

template <class CLASS_T>
void RacerDubinsElevationImpl<CLASS_T>::freeCudaMem()
{
  tex_helper_->freeCudaMem();
  PARENT_CLASS::freeCudaMem();
}

template <class CLASS_T>
void RacerDubinsElevationImpl<CLASS_T>::paramsToDevice()
{
  // does all the internal texture updates
  tex_helper_->copyToDevice();
  PARENT_CLASS::paramsToDevice();
}

template <class CLASS_T>
void RacerDubinsElevationImpl<CLASS_T>::computeParametricModelDeriv(const Eigen::Ref<const state_array>& state,
                                                                    const Eigen::Ref<const control_array>& control,
                                                                    Eigen::Ref<state_array> state_der, const float dt)
{
  float linear_brake_slope = this->params_.c_b[1] / (0.9f * (2 / dt));
  bool enable_brake = control(C_INDEX(THROTTLE_BRAKE)) < 0;
  int index = (abs(state(S_INDEX(VEL_X))) > linear_brake_slope && abs(state(S_INDEX(VEL_X))) <= 3.0) +
              (abs(state(S_INDEX(VEL_X))) > 3.0) * 2;

  state_der(S_INDEX(BRAKE_STATE)) =
      min(max((enable_brake * -control(C_INDEX(THROTTLE_BRAKE)) - state(S_INDEX(BRAKE_STATE))) *
                  this->params_.brake_delay_constant,
              -this->params_.max_brake_rate_neg),
          this->params_.max_brake_rate_pos);
  // applying position throttle
  float throttle = this->params_.c_t[index] * control(C_INDEX(THROTTLE_BRAKE));
  float brake = this->params_.c_b[index] * state(S_INDEX(BRAKE_STATE)) * (state(S_INDEX(VEL_X)) >= 0 ? -1 : 1);
  if (abs(state(S_INDEX(VEL_X))) <= linear_brake_slope)
  {
    throttle = this->params_.c_t[index] * max(control(C_INDEX(THROTTLE_BRAKE)) - this->params_.low_min_throttle, 0.0f);
    brake = linear_brake_slope * state(S_INDEX(BRAKE_STATE)) * -state(S_INDEX(VEL_X));
  }

  state_der(S_INDEX(VEL_X)) = (!enable_brake) * throttle * this->params_.gear_sign + brake -
                              this->params_.c_v[index] * state(S_INDEX(VEL_X)) + this->params_.c_0;
  state_der(S_INDEX(VEL_X)) = min(max(state_der(S_INDEX(VEL_X)), -5.5), 5.5);
  if (abs(state[S_INDEX(PITCH)]) < M_PI_2f32)
  {
    state_der[S_INDEX(VEL_X)] -= this->params_.gravity * sinf(state[S_INDEX(PITCH)]);
  }
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

template <class CLASS_T>
__device__ __host__ void RacerDubinsElevationImpl<CLASS_T>::computeStaticSettling(const float yaw, const float x,
                                                                                  const float y, float roll,
                                                                                  float pitch, float* output)
{
  float height = 0.0f;

  float3 front_left = make_float3(2.981f, 0.737f, 0.0f);
  float3 front_right = make_float3(2.981f, -0.737f, 0.f);
  float3 rear_left = make_float3(0.0f, 0.737f, 0.0f);
  float3 rear_right = make_float3(0.0f, -0.737f, 0.0f);
  front_left = make_float3(front_left.x * cosf(yaw) - front_left.y * sinf(yaw) + x,
                           front_left.x * sinf(yaw) + front_left.y * cosf(yaw) + y, 0.0f);
  front_right = make_float3(front_right.x * cosf(yaw) - front_right.y * sinf(yaw) + x,
                            front_right.x * sinf(yaw) + front_right.y * cosf(yaw) + y, 0.0f);
  rear_left = make_float3(rear_left.x * cosf(yaw) - rear_left.y * sinf(yaw) + x,
                          rear_left.x * sinf(yaw) + rear_left.y * cosf(yaw) + y, 0.0f);
  rear_right = make_float3(rear_right.x * cosf(yaw) - rear_right.y * sinf(yaw) + x,
                           rear_right.x * sinf(yaw) + rear_right.y * cosf(yaw) + y, 0.0f);
  float front_left_height = 0.0f;
  float front_right_height = 0.0f;
  float rear_left_height = 0.0f;
  float rear_right_height = 0.0f;

  if (this->tex_helper_->checkTextureUse(0))
  {
    front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
    front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
    rear_left_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_left);
    rear_right_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_right);

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

  if (isnan(roll) || isinf(roll) || abs(roll) > M_PIf32)
  {
    roll = 4.0f;
  }
  if (isnan(pitch) || isinf(pitch) || abs(pitch) > M_PIf32)
  {
    pitch = 4.0f;
  }
  if (isnan(height) || isinf(height))
  {
    height = 0.0f;
  }

  output[O_INDEX(WHEEL_POS_I_FL_X)] = front_left.x;
  output[O_INDEX(WHEEL_POS_I_FL_Y)] = front_left.y;
  output[O_INDEX(WHEEL_POS_I_FR_X)] = front_right.x;
  output[O_INDEX(WHEEL_POS_I_FR_Y)] = front_right.y;
  output[O_INDEX(WHEEL_POS_I_RL_X)] = rear_left.x;
  output[O_INDEX(WHEEL_POS_I_RL_Y)] = rear_left.y;
  output[O_INDEX(WHEEL_POS_I_RR_X)] = rear_right.x;
  output[O_INDEX(WHEEL_POS_I_RR_Y)] = rear_right.y;
  output[O_INDEX(BASELINK_POS_I_Z)] = height;
}

template <class CLASS_T>
__host__ __device__ void RacerDubinsElevationImpl<CLASS_T>::setOutputs(const float* state_der, const float* next_state,
                                                                       float* output)
{
  // Setup output
  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(PITCH)] = next_state[S_INDEX(PITCH)];
  output[O_INDEX(ROLL)] = next_state[S_INDEX(ROLL)];
  output[O_INDEX(YAW)] = next_state[S_INDEX(YAW)];
  output[O_INDEX(STEER_ANGLE)] = next_state[S_INDEX(STEER_ANGLE)];
  output[O_INDEX(STEER_ANGLE_RATE)] = next_state[S_INDEX(STEER_ANGLE_RATE)];
  output[O_INDEX(WHEEL_FORCE_B_FL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_FR)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RR)] = 10000;
  output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
  output[O_INDEX(ACCEL_Y)] = 0;
  output[O_INDEX(OMEGA_Z)] = state_der[S_INDEX(YAW)];
}

template <class CLASS_T>
void RacerDubinsElevationImpl<CLASS_T>::step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
                                             Eigen::Ref<state_array> state_der,
                                             const Eigen::Ref<const control_array>& control,
                                             Eigen::Ref<output_array> output, const float t, const float dt)
{
  computeParametricModelDeriv(state, control, state_der, dt);

  // Integrate using racer_dubins updateState
  this->PARENT_CLASS::updateState(state, next_state, state_der, dt);

  float pitch = 0;
  float roll = 0;
  computeStaticSettling(state(S_INDEX(YAW)), state(S_INDEX(POS_X)), state(S_INDEX(POS_Y)), roll, pitch, output.data());
  next_state[S_INDEX(PITCH)] = pitch;
  next_state[S_INDEX(ROLL)] = roll;

  setOutputs(state_der.data(), next_state.data(), output.data());
}

template <class CLASS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T>::initializeDynamics(float* state, float* control, float* output,
                                                                      float* theta_s, float t_0, float dt)
{
  PARENT_CLASS::initializeDynamics(state, control, output, theta_s, t_0, dt);
  if (SHARED_MEM_REQUEST_GRD != 1)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* shared_params = (DYN_PARAMS_T*)theta_s;
    *shared_params = this->params_;
  }
}

template <class CLASS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T>::computeParametricModelDeriv(float* state, float* control,
                                                                               float* state_der, const float dt,
                                                                               DYN_PARAMS_T* params_p)
{
  const int tdy = threadIdx.y;
  float linear_brake_slope = params_p->c_b[1] / (0.9f * (2 / dt));

  // Compute dynamics
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  int index = (fabsf(state[S_INDEX(VEL_X)]) > linear_brake_slope && fabsf(state[S_INDEX(VEL_X)]) <= 3.0) +
              (fabsf(state[S_INDEX(VEL_X)]) > 3.0) * 2;

  state_der[S_INDEX(BRAKE_STATE)] =
      min(max((enable_brake * -control[C_INDEX(THROTTLE_BRAKE)] - state[S_INDEX(BRAKE_STATE)]) *
                  params_p->brake_delay_constant,
              -params_p->max_brake_rate_neg),
          params_p->max_brake_rate_pos);

  // applying position throttle
  float throttle = params_p->c_t[index] * control[C_INDEX(THROTTLE_BRAKE)];
  float brake = params_p->c_b[index] * state[S_INDEX(BRAKE_STATE)] * (state[S_INDEX(VEL_X)] >= 0 ? -1 : 1);
  if (abs(state[S_INDEX(VEL_X)]) <= linear_brake_slope)
  {
    throttle = params_p->c_t[index] * max(control[C_INDEX(THROTTLE_BRAKE)] - params_p->low_min_throttle, 0.0f);
    brake = linear_brake_slope * state[S_INDEX(BRAKE_STATE)] * -state[S_INDEX(VEL_X)];
  }

  if (tdy == 0)
  {
    state_der[S_INDEX(VEL_X)] = (!enable_brake) * throttle * params_p->gear_sign + brake -
                                params_p->c_v[index] * state[S_INDEX(VEL_X)] + params_p->c_0;
    state_der[S_INDEX(VEL_X)] = min(max(state_der[S_INDEX(VEL_X)], -5.5), 5.5);
    if (fabsf(state[S_INDEX(PITCH)]) < M_PI_2f32)
    {
      state_der[S_INDEX(VEL_X)] -= params_p->gravity * sinf(state[S_INDEX(PITCH)]);
    }
  }
  state_der[S_INDEX(YAW)] =
      (state[S_INDEX(VEL_X)] / params_p->wheel_base) * tan(state[S_INDEX(STEER_ANGLE)] / params_p->steer_angle_scale);
  state_der[S_INDEX(POS_X)] = state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] = state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min((control[C_INDEX(STEER_CMD)] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
                  params_p->steering_constant,
              params_p->max_steer_rate),
          -params_p->max_steer_rate);
}

template <class CLASS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T>::updateState(float* state, float* next_state, float* state_der,
                                                               const float dt, DYN_PARAMS_T* params_p)
{
  const int tdy = threadIdx.y;
  // Set to 6 as the last 3 states do not do euler integration
  for (int i = tdy; i < 6; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    switch (i)
    {
      case S_INDEX(YAW):
        next_state[i] = angle_utils::normalizeAngle(next_state[i]);
        break;
      case S_INDEX(STEER_ANGLE):
        next_state[S_INDEX(STEER_ANGLE)] =
            max(min(next_state[S_INDEX(STEER_ANGLE)], params_p->max_steer_angle), -params_p->max_steer_angle);
        next_state[S_INDEX(STEER_ANGLE_RATE)] = state_der[S_INDEX(STEER_ANGLE)];
        break;
      case S_INDEX(BRAKE_STATE):
        next_state[S_INDEX(BRAKE_STATE)] =
            min(max(next_state[S_INDEX(BRAKE_STATE)], 0.0f), -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x);
        break;
      }
    }
    __syncthreads();
}

template <class CLASS_T>
__device__ inline void RacerDubinsElevationImpl<CLASS_T>::step(float* state, float* next_state, float* state_der,
                                                               float* control, float* output, float* theta_s,
                                                               const float t, const float dt)
{
  DYN_PARAMS_T* params_p;
  if (SHARED_MEM_REQUEST_GRD != 1)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta_s;
  }
  else
  {
    params_p = &(this->params_);
  }
  computeParametricModelDeriv(state, control, state_der, dt, params_p);

  float pitch = 0;
  float roll = 0;

  if (threadIdx.y == 7)
  {
    computeStaticSettling(state[S_INDEX(YAW)], state[S_INDEX(POS_X)], state[S_INDEX(POS_Y)], roll, pitch, output);
    next_state[S_INDEX(PITCH)] = pitch;
    next_state[S_INDEX(ROLL)] = roll;
  }

  updateState(state, next_state, state_der, dt, params_p);
  setOutputs(state_der, next_state, output);
}

template <class CLASS_T>
RacerDubinsElevationImpl<CLASS_T>::state_array
RacerDubinsElevationImpl<CLASS_T>::stateFromMap(const std::map<std::string, float>& map)
{
  state_array s = state_array::Zero();
  if (map.find("VEL_X") == map.end() || map.find("VEL_Y") == map.end() || map.find("POS_X") == map.end() ||
      map.find("POS_Y") == map.end() || map.find("ROLL") == map.end() || map.find("PITCH") == map.end() ||
      map.find("STEER_ANGLE") == map.end() || map.find("STEER_ANGLE_RATE") == map.end() ||
      map.find("BRAKE_STATE") == map.end())
  {
    std::cout << "WARNING: could not find all keys for elevation dynamics" << std::endl;
    for (const auto& it : map)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return s;
  }
  s(S_INDEX(POS_X)) = map.at("POS_X");
  s(S_INDEX(POS_Y)) = map.at("POS_Y");
  s(S_INDEX(VEL_X)) = map.at("VEL_X");
  s(S_INDEX(YAW)) = map.at("YAW");
  s(S_INDEX(STEER_ANGLE)) = map.at("STEER_ANGLE");
  s(S_INDEX(STEER_ANGLE_RATE)) = map.at("STEER_ANGLE_RATE");
  s(S_INDEX(ROLL)) = map.at("ROLL");
  s(S_INDEX(PITCH)) = map.at("PITCH");
  s(S_INDEX(BRAKE_STATE)) = map.at("BRAKE_STATE");
  return s;
}
