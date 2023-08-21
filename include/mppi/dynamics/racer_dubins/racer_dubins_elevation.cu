#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>
#include <mppi/utils/math_utils.h>

template <class CLASS_T, class PARAMS_T>
void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::GPUSetup()
{
  PARENT_CLASS::GPUSetup();
  tex_helper_->GPUSetup();
  // makes sure that the device ptr sees the correct texture object
  HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->tex_helper_), &(tex_helper_->ptr_d_),
                               sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::freeCudaMem()
{
  tex_helper_->freeCudaMem();
  PARENT_CLASS::freeCudaMem();
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::paramsToDevice()
{
  // does all the internal texture updates
  tex_helper_->copyToDevice();
  PARENT_CLASS::paramsToDevice();
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::computeParametricDelayDeriv(
    const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    Eigen::Ref<state_array> state_der)
{
  bool enable_brake = control(C_INDEX(THROTTLE_BRAKE)) < 0.0f;

  state_der(S_INDEX(BRAKE_STATE)) =
      min(max((enable_brake * -control(C_INDEX(THROTTLE_BRAKE)) - state(S_INDEX(BRAKE_STATE))) *
                  this->params_.brake_delay_constant,
              -this->params_.max_brake_rate_neg),
          this->params_.max_brake_rate_pos);
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::computeParametricSteerDeriv(
    const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    Eigen::Ref<state_array> state_der)
{
  state_der(S_INDEX(STEER_ANGLE)) =
      (control(C_INDEX(STEER_CMD)) * this->params_.steer_command_angle_scale - state(S_INDEX(STEER_ANGLE))) *
      this->params_.steering_constant;
  state_der(S_INDEX(STEER_ANGLE)) =
      max(min(state_der(S_INDEX(STEER_ANGLE)), this->params_.max_steer_rate), -this->params_.max_steer_rate);
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::computeParametricAccelDeriv(
    const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    Eigen::Ref<state_array> state_der, const float dt)
{
  float linear_brake_slope = this->params_.c_b[1] / (0.9f * (2.0f / dt));
  bool enable_brake = control(C_INDEX(THROTTLE_BRAKE)) < 0.0f;
  int index = (abs(state(S_INDEX(VEL_X))) > linear_brake_slope && abs(state(S_INDEX(VEL_X))) <= 3.0f) +
              (abs(state(S_INDEX(VEL_X))) > 3.0f) * 2;

  // applying position throttle
  float throttle = this->params_.c_t[index] * control(C_INDEX(THROTTLE_BRAKE));
  float brake = this->params_.c_b[index] * state(S_INDEX(BRAKE_STATE)) * (state(S_INDEX(VEL_X)) >= 0.0f ? -1.0f : 1.0f);
  if (abs(state(S_INDEX(VEL_X))) <= linear_brake_slope)
  {
    throttle = this->params_.c_t[index] * max(control(C_INDEX(THROTTLE_BRAKE)) - this->params_.low_min_throttle, 0.0f);
    brake = this->params_.c_b[index] * state(S_INDEX(BRAKE_STATE)) * -state(S_INDEX(VEL_X));
  }

  state_der(S_INDEX(VEL_X)) = (!enable_brake) * throttle * this->params_.gear_sign + brake -
                              this->params_.c_v[index] * state(S_INDEX(VEL_X)) + this->params_.c_0;
  state_der(S_INDEX(VEL_X)) = min(max(state_der(S_INDEX(VEL_X)), -5.5f), 5.5f);
  if (abs(state[S_INDEX(PITCH)]) < M_PI_2f32)
  {
    state_der[S_INDEX(VEL_X)] -= this->params_.gravity * sinf(state[S_INDEX(PITCH)]);
  }
  state_der(S_INDEX(YAW)) = (state(S_INDEX(VEL_X)) / this->params_.wheel_base) *
                            tanf(state(S_INDEX(STEER_ANGLE)) / this->params_.steer_angle_scale);
  state_der(S_INDEX(POS_X)) = state(S_INDEX(VEL_X)) * cosf(state(S_INDEX(YAW)));
  state_der(S_INDEX(POS_Y)) = state(S_INDEX(VEL_X)) * sinf(state(S_INDEX(YAW)));
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::setOutputs(const float* state_der,
                                                                                 const float* next_state, float* output)
{
  // Setup output
  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = 0.0f;
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0.0f;
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(PITCH)] = next_state[S_INDEX(PITCH)];
  output[O_INDEX(ROLL)] = next_state[S_INDEX(ROLL)];
  output[O_INDEX(YAW)] = next_state[S_INDEX(YAW)];
  output[O_INDEX(STEER_ANGLE)] = next_state[S_INDEX(STEER_ANGLE)];
  output[O_INDEX(STEER_ANGLE_RATE)] = next_state[S_INDEX(STEER_ANGLE_RATE)];
  output[O_INDEX(WHEEL_FORCE_B_FL)] = 10000.0f;
  output[O_INDEX(WHEEL_FORCE_B_FR)] = 10000.0f;
  output[O_INDEX(WHEEL_FORCE_B_RL)] = 10000.0f;
  output[O_INDEX(WHEEL_FORCE_B_RR)] = 10000.0f;
  output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
  output[O_INDEX(ACCEL_Y)] = 0.0f;
  output[O_INDEX(OMEGA_Z)] = state_der[S_INDEX(YAW)];
}

template <class CLASS_T, class PARAMS_T>
void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::step(Eigen::Ref<state_array> state,
                                                       Eigen::Ref<state_array> next_state,
                                                       Eigen::Ref<state_array> state_der,
                                                       const Eigen::Ref<const control_array>& control,
                                                       Eigen::Ref<output_array> output, const float t, const float dt)
{
  computeParametricDelayDeriv(state, control, state_der);
  computeParametricSteerDeriv(state, control, state_der);
  computeParametricAccelDeriv(state, control, state_der, dt);

  // Integrate using racer_dubins updateState
  this->PARENT_CLASS::updateState(state, next_state, state_der, dt);

  float roll = state(S_INDEX(ROLL));
  float pitch = state(S_INDEX(PITCH));
  RACER::computeStaticSettling<typename DYN_PARAMS_T::OutputIndex, TwoDTextureHelper<float>>(
      this->tex_helper_, next_state(S_INDEX(YAW)), next_state(S_INDEX(POS_X)), next_state(S_INDEX(POS_Y)), roll, pitch,
      output.data());
  next_state[S_INDEX(PITCH)] = pitch;
  next_state[S_INDEX(ROLL)] = roll;

  setOutputs(state_der.data(), next_state.data(), output.data());
}

template <class CLASS_T, class PARAMS_T>
bool RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::computeGrad(const Eigen::Ref<const state_array>& state,
                                                              const Eigen::Ref<const control_array>& control,
                                                              Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B)
{
  A = dfdx::Zero();
  B = dfdu::Zero();

  float eps = 0.01f;
  bool enable_brake = control(C_INDEX(THROTTLE_BRAKE)) < 0.0f;

  // vx
  float linear_brake_slope = this->params_.c_b[1] / (0.9f * (2.0f / 0.01));
  int index = (abs(state(S_INDEX(VEL_X))) > linear_brake_slope && abs(state(S_INDEX(VEL_X))) <= 3.0f) +
              (abs(state(S_INDEX(VEL_X))) > 3.0f) * 2;

  A(0, 0) = -this->params_.c_v[index];
  if (abs(state(S_INDEX(VEL_X))) < linear_brake_slope)
  {
    A(0, 5) = this->params_.c_b[index] * -state(S_INDEX(VEL_X));
  }
  else
  {
    A(0, 5) = this->params_.c_b[index] * (state(S_INDEX(VEL_X)) >= 0.0f ? -1.0f : 1.0f);
  }
  // TODO zero out if we are above the threshold to match??

  // yaw
  A(1, 0) = (1.0f / this->params_.wheel_base) * tanf(state(S_INDEX(STEER_ANGLE)) / this->params_.steer_angle_scale);
  A(1, 4) = (state(S_INDEX(VEL_X)) / this->params_.wheel_base) *
            (1.0 / SQ(cosf(state(S_INDEX(STEER_ANGLE)) / this->params_.steer_angle_scale))) /
            this->params_.steer_angle_scale;
  // pos x
  A(2, 0) = cosf(state(S_INDEX(YAW)));
  A(2, 1) = -sinf(state(S_INDEX(YAW))) * state(S_INDEX(VEL_X));
  // pos y
  A(3, 0) = sinf(state(S_INDEX(YAW)));
  A(3, 1) = cosf(state(S_INDEX(YAW))) * state(S_INDEX(VEL_X));
  // steer angle
  float steer_dot =
      (control(C_INDEX(STEER_CMD)) * this->params_.steer_command_angle_scale - state(S_INDEX(STEER_ANGLE))) *
      this->params_.steering_constant;
  if (steer_dot - eps < -this->params_.max_steer_rate || steer_dot + eps > this->params_.max_steer_rate)
  {
    A(4, 4) = 0.0f;
  }
  else
  {
    A(4, 4) = -this->params_.steering_constant;
  }
  A(4, 4) = max(min(A(4, 4), this->params_.max_steer_rate), -this->params_.max_steer_rate);
  // gravity
  A(0, 7) = -this->params_.gravity * cosf(state(S_INDEX(PITCH)));

  // brake delay
  float brake_dot = (enable_brake * -control(C_INDEX(THROTTLE_BRAKE)) - state(S_INDEX(BRAKE_STATE))) *
                    this->params_.brake_delay_constant;
  if (brake_dot - eps < -this->params_.max_brake_rate_neg || brake_dot + eps > this->params_.max_brake_rate_pos)
  {
    A(5, 5) = 0.0f;
  }
  else
  {
    A(5, 5) = -this->params_.brake_delay_constant;
  }

  // steering command
  B(4, 1) = this->params_.steer_command_angle_scale * this->params_.steering_constant;
  // throttle command
  B(0, 0) = this->params_.c_t[index] * this->params_.gear_sign * (!enable_brake);
  // brake command
  if ((state(S_INDEX(BRAKE_STATE)) < -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x && brake_dot < 0.0f) ||
      (state(S_INDEX(BRAKE_STATE)) > 0.0f && brake_dot > 0.0f))
  {
    B(5, 0) = -this->params_.brake_delay_constant * enable_brake;
  }
  return true;
}

template <class CLASS_T, class PARAMS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::initializeDynamics(float* state, float* control,
                                                                                float* output, float* theta_s,
                                                                                float t_0, float dt)
{
  PARENT_CLASS::initializeDynamics(state, control, output, theta_s, t_0, dt);
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* shared_params = (DYN_PARAMS_T*)theta_s;
    *shared_params = this->params_;
  }
}

template <class CLASS_T, class PARAMS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::computeParametricDelayDeriv(float* state, float* control,
                                                                                         float* state_der,
                                                                                         DYN_PARAMS_T* params_p)
{
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0.0f;

  // Compute dynamics
  state_der[S_INDEX(BRAKE_STATE)] =
      min(max((enable_brake * -control[C_INDEX(THROTTLE_BRAKE)] - state[S_INDEX(BRAKE_STATE)]) *
                  params_p->brake_delay_constant,
              -params_p->max_brake_rate_neg),
          params_p->max_brake_rate_pos);
}

template <class CLASS_T, class PARAMS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::computeParametricSteerDeriv(float* state, float* control,
                                                                                         float* state_der,
                                                                                         DYN_PARAMS_T* params_p)
{
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min((control[C_INDEX(STEER_CMD)] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
                  params_p->steering_constant,
              params_p->max_steer_rate),
          -params_p->max_steer_rate);
}

template <class CLASS_T, class PARAMS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::computeParametricAccelDeriv(float* state, float* control,
                                                                                         float* state_der,
                                                                                         const float dt,
                                                                                         DYN_PARAMS_T* params_p)
{
  const int tdy = threadIdx.y;
  float linear_brake_slope = params_p->c_b[1] / (0.9f * (2.0f / dt));

  // Compute dynamics
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0.0f;
  int index = (fabsf(state[S_INDEX(VEL_X)]) > linear_brake_slope && fabsf(state[S_INDEX(VEL_X)]) <= 3.0f) +
              (fabsf(state[S_INDEX(VEL_X)]) > 3.0f) * 2;
  // applying position throttle
  float throttle = params_p->c_t[index] * control[C_INDEX(THROTTLE_BRAKE)];
  float brake = params_p->c_b[index] * state[S_INDEX(BRAKE_STATE)] * (state[S_INDEX(VEL_X)] >= 0.0f ? -1.0f : 1.0f);
  if (fabsf(state[S_INDEX(VEL_X)]) <= linear_brake_slope)
  {
    throttle = params_p->c_t[index] * max(control[C_INDEX(THROTTLE_BRAKE)] - params_p->low_min_throttle, 0.0f);
    brake = params_p->c_b[index] * state[S_INDEX(BRAKE_STATE)] * -state[S_INDEX(VEL_X)];
  }

  if (tdy == 0)
  {
    state_der[S_INDEX(VEL_X)] = (!enable_brake) * throttle * params_p->gear_sign + brake -
                                params_p->c_v[index] * state[S_INDEX(VEL_X)] + params_p->c_0;
    state_der[S_INDEX(VEL_X)] = min(max(state_der[S_INDEX(VEL_X)], -5.5f), 5.5f);
    if (fabsf(state[S_INDEX(PITCH)]) < M_PI_2f32)
    {
      state_der[S_INDEX(VEL_X)] -= params_p->gravity * __sinf(angle_utils::normalizeAngle(state[S_INDEX(PITCH)]));
    }
  }
  state_der[S_INDEX(YAW)] =
      (state[S_INDEX(VEL_X)] / params_p->wheel_base) *
      __tanf(angle_utils::normalizeAngle(state[S_INDEX(STEER_ANGLE)] / params_p->steer_angle_scale));
  const float yaw_norm = angle_utils::normalizeAngle(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_X)] = state[S_INDEX(VEL_X)] * __cosf(yaw_norm);
  state_der[S_INDEX(POS_Y)] = state[S_INDEX(VEL_X)] * __sinf(yaw_norm);
}

template <class CLASS_T, class PARAMS_T>
__device__ void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::updateState(float* state, float* next_state,
                                                                         float* state_der, const float dt,
                                                                         DYN_PARAMS_T* params_p)
{
  const uint tdy = threadIdx.y;
  // Set to 6 as the last 3 states do not do euler integration
  for (uint i = tdy; i < 6; i += blockDim.y)
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

template <class CLASS_T, class PARAMS_T>
__device__ inline void RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::step(float* state, float* next_state,
                                                                         float* state_der, float* control,
                                                                         float* output, float* theta_s, const float t,
                                                                         const float dt)
{
  DYN_PARAMS_T* params_p;
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta_s;
  }
  else
  {
    params_p = &(this->params_);
  }
  computeParametricDelayDeriv(state, control, state_der, params_p);
  computeParametricSteerDeriv(state, control, state_der, params_p);
  computeParametricAccelDeriv(state, control, state_der, dt, params_p);

  updateState(state, next_state, state_der, dt, params_p);

  if (threadIdx.y == 0)
  {
    float roll = state[S_INDEX(ROLL)];
    float pitch = state[S_INDEX(PITCH)];
    RACER::computeStaticSettling<DYN_PARAMS_T::OutputIndex, TwoDTextureHelper<float>>(
        this->tex_helper_, next_state[S_INDEX(YAW)], next_state[S_INDEX(POS_X)], next_state[S_INDEX(POS_Y)], roll,
        pitch, output);
    next_state[S_INDEX(PITCH)] = pitch;
    next_state[S_INDEX(ROLL)] = roll;
  }
  setOutputs(state_der, next_state, output);
}

template <class CLASS_T, class PARAMS_T>
RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::state_array
RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::stateFromMap(const std::map<std::string, float>& map)
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

template <class CLASS_T, class PARAMS_T>
Eigen::Quaternionf
RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::attitudeFromState(const Eigen::Ref<const state_array>& state)
{
  Eigen::Quaternionf q;
  mppi::math::Euler2QuatNWU(state(S_INDEX(ROLL)), state(S_INDEX(PITCH)), state(S_INDEX(YAW)), q);
  return q;
}

template <class CLASS_T, class PARAMS_T>
Eigen::Vector3f
RacerDubinsElevationImpl<CLASS_T, PARAMS_T>::positionFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(POS_X)], state[S_INDEX(POS_Y)], 0.0f);
}
