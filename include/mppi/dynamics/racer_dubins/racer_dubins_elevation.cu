#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>
#include <mppi/utils/math_utils.h>

void RacerDubinsElevation::GPUSetup()
{
  PARENT_CLASS* derived = static_cast<PARENT_CLASS*>(this);
  CudaCheckError();
  tex_helper_->GPUSetup();
  CudaCheckError();
  derived->GPUSetup();
  CudaCheckError();
}

void RacerDubinsElevation::freeCudaMem()
{
  tex_helper_->freeCudaMem();
}

void RacerDubinsElevation::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    // does all the internal texture updates
    tex_helper_->copyToDevice();
    // makes sure that the device ptr sees the correct texture object
    HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->tex_helper_), &(tex_helper_->ptr_d_),
                                 sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
  }
  PARENT_CLASS::paramsToDevice();
}

void RacerDubinsElevation::updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                                       Eigen::Ref<state_array> state_der, const float dt)
{
  next_state = state + state_der * dt;
  next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
  next_state(S_INDEX(STEER_ANGLE)) =
      max(min(next_state(S_INDEX(STEER_ANGLE)), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  next_state(7) = state_der(S_INDEX(STEER_ANGLE));
  next_state(S_INDEX(ACCEL_X)) = state_der(S_INDEX(VEL_X));
}

__device__ void RacerDubinsElevation::updateState(float* state, float* next_state, float* state_der, const float dt)
{
  int i;
  int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < 5; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    if (i == S_INDEX(VEL_X))
    {
      next_state[S_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
    }
    if (i == S_INDEX(YAW))
    {
      next_state[i] = angle_utils::normalizeAngle(next_state[i]);
    }
    if (i == S_INDEX(STEER_ANGLE))
    {
      next_state[S_INDEX(STEER_ANGLE)] =
          max(min(next_state[S_INDEX(STEER_ANGLE)], this->params_.max_steer_angle), -this->params_.max_steer_angle);
      next_state[S_INDEX(STEER_ANGLE_RATE)] = state_der[S_INDEX(STEER_ANGLE)];
    }
  }
}

void RacerDubinsElevation::computeStateDeriv(const Eigen::Ref<const state_array>& state,
                                             const Eigen::Ref<const control_array>& control,
                                             Eigen::Ref<state_array> state_der)
{
}

void RacerDubinsElevation::step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
                                Eigen::Ref<state_array> state_der, const Eigen::Ref<const control_array>& control,
                                Eigen::Ref<output_array> output, const float t, const float dt)
{
  // computeStateDeriv(state, control, state_der);
  bool enable_brake = control(0) < 0;
  int index = (abs(state(0)) > 0.5 && abs(state(0)) <= 6.0) + (abs(state(0)) > 6.0) * 2;
  // applying position throttle
  float throttle = this->params_.c_t[index] * control(0);
  float brake = this->params_.c_b[index] * control(0) * (state(0) >= 0 ? 1 : -1);
  if (abs(state(0)) <= 0.5)
  {
    throttle = this->params_.c_t[index] * max(control(0) - this->params_.low_min_throttle, 0.0f);
    brake = this->params_.c_b[index] * control(0) * state(0);
  }

  state_der(0) =
      (!enable_brake) * throttle + (enable_brake)*brake - this->params_.c_v[index] * state(0) + this->params_.c_0;
  if (abs(state[6]) < M_PI_2)
  {
    state_der[0] -= this->params_.gravity * sinf(state[6]);
  }
  state_der(1) = (state(0) / this->params_.wheel_base) * tan(state(4) / this->params_.steer_angle_scale[index]);
  state_der(2) = state(0) * cosf(state(1));
  state_der(3) = state(0) * sinf(state(1));
  state_der(4) = (control(1) * this->params_.steer_command_angle_scale - state(4)) * this->params_.steering_constant;
  state_der(4) = max(min(state_der(4), this->params_.max_steer_rate), -this->params_.max_steer_rate);
  // state(8) = state_der(0);

  // Integrate using racer_dubins upddateState
  updateState(state, next_state, state_der, dt);

  float pitch = 0;
  float roll = 0;

  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);
  front_left = make_float3(front_left.x * cosf(state(1)) - front_left.y * sinf(state(1)) + state(2),
                           front_left.x * sinf(state(1)) + front_left.y * cosf(state(1)) + state(3), 0);
  front_right = make_float3(front_right.x * cosf(state(1)) - front_right.y * sinf(state(1)) + state(2),
                            front_right.x * sinf(state(1)) + front_right.y * cosf(state(1)) + state(3), 0);
  rear_left = make_float3(rear_left.x * cosf(state(1)) - rear_left.y * sinf(state(1)) + state(2),
                          rear_left.x * sinf(state(1)) + rear_left.y * cosf(state(1)) + state(3), 0);
  rear_right = make_float3(rear_right.x * cosf(state(1)) - rear_right.y * sinf(state(1)) + state(2),
                           rear_right.x * sinf(state(1)) + rear_right.y * cosf(state(1)) + state(3), 0);
  float front_left_height = 0;
  float front_right_height = 0;
  float rear_left_height = 0;
  float rear_right_height = 0;

  if (this->tex_helper_->checkTextureUse(0))
  {
    front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
    front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
    rear_left_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_left);
    rear_right_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_right);

    float front_diff = front_left_height - front_right_height;
    front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
    float rear_diff = rear_left_height - rear_right_height;
    rear_diff = max(min(rear_diff, 0.736 * 2), -0.736 * 2);
    float front_roll = asinf(front_diff / (0.737 * 2));
    float rear_roll = asinf(rear_diff / (0.737 * 2));
    next_state(5) = (front_roll + rear_roll) / 2;

    float left_diff = rear_left_height - front_left_height;
    left_diff = max(min(left_diff, 2.98), -2.98);
    float right_diff = rear_right_height - front_right_height;
    right_diff = max(min(right_diff, 2.98), -2.98);
    float left_pitch = asinf((left_diff) / 2.981);
    float right_pitch = asinf((right_diff) / 2.981);
    next_state(6) = (left_pitch + right_pitch) / 2;
  }
  else
  {
    next_state(5) = 0;
    next_state(6) = 0;
  }

  if (isnan(next_state(5)) || isinf(next_state(5)) || abs(next_state(5)) > M_PI)
  {
    next_state(5) = 4.0;
  }
  if (isnan(next_state(6)) || isinf(next_state(6)) || abs(next_state(6)) > M_PI)
  {
    next_state(6) = 4.0;
  }
  // state_der[5] = (next_state[5] - state[5]) / dt;
  // state_der[6] = (next_state[6] - state[6]) / dt;

  // Setup output
  float yaw = next_state[S_INDEX(YAW)];
  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(BASELINK_POS_I_Z)] = 0;
  output[O_INDEX(OMEGA_B_X)] = 0;
  output[O_INDEX(OMEGA_B_Y)] = 0;
  output[O_INDEX(OMEGA_B_Z)] = 0;
  output[O_INDEX(YAW)] = yaw;
  output[O_INDEX(PITCH)] = pitch;
  output[O_INDEX(ROLL)] = roll;
  Eigen::Quaternionf q;
  mppi::math::Euler2QuatNWU(roll, pitch, yaw, q);
  output[O_INDEX(ATTITUDE_QW)] = q.w();
  output[O_INDEX(ATTITUDE_QX)] = q.x();
  output[O_INDEX(ATTITUDE_QY)] = q.y();
  output[O_INDEX(ATTITUDE_QZ)] = q.z();
  output[O_INDEX(STEER_ANGLE)] = next_state[S_INDEX(STEER_ANGLE)];
  output[O_INDEX(STEER_ANGLE_RATE)] = 0;
  output[O_INDEX(WHEEL_POS_I_FL_X)] = front_left.x;
  output[O_INDEX(WHEEL_POS_I_FL_Y)] = front_left.y;
  output[O_INDEX(WHEEL_POS_I_FR_X)] = front_right.x;
  output[O_INDEX(WHEEL_POS_I_FR_Y)] = front_right.y;
  output[O_INDEX(WHEEL_POS_I_RL_X)] = rear_left.x;
  output[O_INDEX(WHEEL_POS_I_RL_Y)] = rear_left.y;
  output[O_INDEX(WHEEL_POS_I_RR_X)] = rear_right.x;
  output[O_INDEX(WHEEL_POS_I_RR_Y)] = rear_right.y;
  output[O_INDEX(WHEEL_FORCE_B_FL_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FL_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FL_Z)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_FR_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FR_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FR_Z)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RL_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RL_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RL_Z)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RR_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RR_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RR_Z)] = 10000;
  output[O_INDEX(CENTER_POS_I_X)] = output[O_INDEX(BASELINK_POS_I_X)];  // TODO
  output[O_INDEX(CENTER_POS_I_Y)] = output[O_INDEX(BASELINK_POS_I_Y)];
  output[O_INDEX(CENTER_POS_I_Z)] = 0;
}

__device__ inline void RacerDubinsElevation::step(float* state, float* next_state, float* state_der, float* control,
                                                  float* output, float* theta_s, const float t, const float dt)
{
  // computeStateDeriv(state, control, state_der, theta_s);
  bool enable_brake = control[0] < 0;
  int index =
      (abs(state[S_INDEX(VEL_X)]) > 0.5 && abs(state[S_INDEX(VEL_X)]) <= 6.0) + (abs(state[S_INDEX(VEL_X)]) > 6.0) * 2;
  // applying position throttle
  float throttle = this->params_.c_t[index] * control[0];
  float brake = this->params_.c_b[index] * control[0] * (state[S_INDEX(VEL_X)] >= 0 ? 1 : -1);
  if (abs(state[S_INDEX(VEL_X)]) <= 0.5)
  {
    throttle = this->params_.c_t[index] * max(control[0] - this->params_.low_min_throttle, 0.0f);
    brake = this->params_.c_b[index] * control[0] * state[S_INDEX(VEL_X)];
  }

  state_der[S_INDEX(VEL_X)] = (!enable_brake) * throttle + (enable_brake)*brake -
                              this->params_.c_v[index] * state[S_INDEX(VEL_X)] + this->params_.c_0;
  if (abs(state[S_INDEX(PITCH)]) < M_PI_2)
  {
    state_der[S_INDEX(VEL_X)] -= this->params_.gravity * sinf(state[S_INDEX(PITCH)]);
  }
  state_der[S_INDEX(YAW)] = (state[S_INDEX(VEL_X)] / this->params_.wheel_base) *
                            tan(state[S_INDEX(STEER_ANGLE)] / this->params_.steer_angle_scale[index]);
  state_der[S_INDEX(POS_X)] = state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] = state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(STEER_ANGLE)] =
      (control[1] * this->params_.steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
      this->params_.steering_constant;
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min(state_der[S_INDEX(STEER_ANGLE)], this->params_.max_steer_rate), -this->params_.max_steer_rate);
  __syncthreads();
  // Use Euler Integration from racer_dubins parent class
  updateState(state, next_state, state_der, dt);
  __syncthreads();

  float pitch = 0;
  float roll = 0;

  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);
  front_left = make_float3(front_left.x * cosf(state[1]) - front_left.y * sinf(state[1]) + state[2],
                           front_left.x * sinf(state[1]) + front_left.y * cosf(state[1]) + state[3], 0);
  front_right = make_float3(front_right.x * cosf(state[1]) - front_right.y * sinf(state[1]) + state[2],
                            front_right.x * sinf(state[1]) + front_right.y * cosf(state[1]) + state[3], 0);
  rear_left = make_float3(rear_left.x * cosf(state[1]) - rear_left.y * sinf(state[1]) + state[2],
                          rear_left.x * sinf(state[1]) + rear_left.y * cosf(state[1]) + state[3], 0);
  rear_right = make_float3(rear_right.x * cosf(state[1]) - rear_right.y * sinf(state[1]) + state[2],
                           rear_right.x * sinf(state[1]) + rear_right.y * cosf(state[1]) + state[3], 0);
  float front_left_height = 0;
  float front_right_height = 0;
  float rear_left_height = 0;
  float rear_right_height = 0;

  if (this->tex_helper_->checkTextureUse(0))
  {
    front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
    front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
    rear_left_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_left);
    rear_right_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_right);

    // max magnitude
    float front_diff = front_left_height - front_right_height;
    front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
    float rear_diff = rear_left_height - rear_right_height;
    rear_diff = max(min(rear_diff, 0.736 * 2), -0.736 * 2);
    float front_roll = asinf(front_diff / (0.737 * 2));
    float rear_roll = asinf(rear_diff / (0.737 * 2));
    next_state[5] = (front_roll + rear_roll) / 2;

    float left_diff = rear_left_height - front_left_height;
    left_diff = max(min(left_diff, 2.98), -2.98);
    float right_diff = rear_right_height - front_right_height;
    right_diff = max(min(right_diff, 2.98), -2.98);
    float left_pitch = asinf((left_diff) / 2.981);
    float right_pitch = asinf((right_diff) / 2.981);
    next_state[6] = (left_pitch + right_pitch) / 2;
  }
  else
  {
    next_state[5] = 0;
    next_state[6] = 0;
  }

  if (isnan(next_state[5]) || isinf(next_state[5]) || abs(next_state[5]) > M_PI)
  {
    next_state[5] = 4.0;
  }
  if (isnan(next_state[6]) || isinf(next_state[6]) || abs(next_state[6]) > M_PI)
  {
    next_state[6] = 4.0;
  }
  // state_der[5] = (next_state[5] - state[5]) / dt;
  // state_der[6] = (next_state[6] - state[6]) / dt;
  if (output)
  {
    float yaw = next_state[S_INDEX(YAW)];
    output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
    output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
    output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
    output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
    output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
    output[O_INDEX(BASELINK_POS_I_Z)] = 0;
    output[O_INDEX(OMEGA_B_X)] = 0;
    output[O_INDEX(OMEGA_B_Y)] = 0;
    output[O_INDEX(OMEGA_B_Z)] = 0;
    output[O_INDEX(YAW)] = yaw;
    output[O_INDEX(PITCH)] = pitch;
    output[O_INDEX(ROLL)] = roll;
    Eigen::Quaternionf q;
    mppi::math::Euler2QuatNWU(roll, pitch, yaw, q);
    output[O_INDEX(ATTITUDE_QW)] = q.w();
    output[O_INDEX(ATTITUDE_QX)] = q.x();
    output[O_INDEX(ATTITUDE_QY)] = q.y();
    output[O_INDEX(ATTITUDE_QZ)] = q.z();
    output[O_INDEX(STEER_ANGLE)] = next_state[S_INDEX(STEER_ANGLE)];
    output[O_INDEX(STEER_ANGLE_RATE)] = 0;
    output[O_INDEX(WHEEL_POS_I_FL_X)] = front_left.x;
    output[O_INDEX(WHEEL_POS_I_FL_Y)] = front_left.y;
    output[O_INDEX(WHEEL_POS_I_FR_X)] = front_right.x;
    output[O_INDEX(WHEEL_POS_I_FR_Y)] = front_right.y;
    output[O_INDEX(WHEEL_POS_I_RL_X)] = rear_left.x;
    output[O_INDEX(WHEEL_POS_I_RL_Y)] = rear_left.y;
    output[O_INDEX(WHEEL_POS_I_RR_X)] = rear_right.x;
    output[O_INDEX(WHEEL_POS_I_RR_Y)] = rear_right.y;
    output[O_INDEX(WHEEL_FORCE_B_FL_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FL_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FL_Z)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_FR_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FR_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FR_Z)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_RL_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RL_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RL_Z)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_RR_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RR_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RR_Z)] = 10000;
    output[O_INDEX(CENTER_POS_I_X)] = output[O_INDEX(BASELINK_POS_I_X)];  // TODO
    output[O_INDEX(CENTER_POS_I_Y)] = output[O_INDEX(BASELINK_POS_I_Y)];
    output[O_INDEX(CENTER_POS_I_Z)] = 0;
  }
}
