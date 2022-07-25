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

__device__ void RacerDubinsElevation::initializeDynamics(float* state, float* control, float* theta_s, float t_0,
                                                         float dt)
{
  if (SHARED_MEM_REQUEST_GRD != 1)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* shared_params = (DYN_PARAMS_T*)theta_s;
    *shared_params = this->params_;
  }
}

__device__ inline void RacerDubinsElevation::step(float* state, float* next_state, float* state_der, float* control,
                                                  float* output, float* theta_s, const float t, const float dt)
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
  // computeStateDeriv(state, control, state_der, theta_s);
  const int tdy = threadIdx.y;
  // if (tdy == 1 % blockDim.y)
  // { // calculate on separate thread from cost to potentially have this happen in conjunction
  bool enable_brake = control[0] < 0;
  int index = (fabsf(state[S_INDEX(VEL_X)]) > 0.5 && fabsf(state[S_INDEX(VEL_X)]) <= 6.0) +
              (fabsf(state[S_INDEX(VEL_X)]) > 6.0) * 2;
  // applying position throttle
  float throttle = params_p->c_t[index] * control[0];
  float brake = params_p->c_b[index] * control[0] * (state[S_INDEX(VEL_X)] >= 0 ? 1 : -1);
  if (abs(state[S_INDEX(VEL_X)]) <= 0.5)
  {
    throttle = params_p->c_t[index] * max(control[0] - params_p->low_min_throttle, 0.0f);
    brake = params_p->c_b[index] * control[0] * state[S_INDEX(VEL_X)];
  }

  state_der[S_INDEX(VEL_X)] =
      (!enable_brake) * throttle + (enable_brake)*brake - params_p->c_v[index] * state[S_INDEX(VEL_X)] + params_p->c_0;
  if (fabsf(state[S_INDEX(PITCH)]) < M_PI_2f32)
  {
    state_der[S_INDEX(VEL_X)] -= params_p->gravity * sinf(state[S_INDEX(PITCH)]);
  }
  state_der[S_INDEX(YAW)] = (state[S_INDEX(VEL_X)] / params_p->wheel_base) *
                            tan(state[S_INDEX(STEER_ANGLE)] / params_p->steer_angle_scale[index]);
  state_der[S_INDEX(POS_X)] = state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] = state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(STEER_ANGLE)] =
      (control[1] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) * params_p->steering_constant;
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min(state_der[S_INDEX(STEER_ANGLE)], params_p->max_steer_rate), -params_p->max_steer_rate);
  // }
  __syncthreads();
  // Use Euler Integration from racer_dubins parent class
  // updateState(state, next_state, state_der, dt);
  float pitch = 0;
  float roll = 0;

  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);

  float front_left_height = 0;
  float front_right_height = 0;
  float rear_left_height = 0;
  float rear_right_height = 0;
  for (int i = tdy; i < 5; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    switch (i)
    {
      case S_INDEX(VEL_X):
        next_state[S_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
        break;
      case S_INDEX(YAW):
        next_state[i] = angle_utils::normalizeAngle(next_state[i]);
        break;
      case S_INDEX(STEER_ANGLE):
        next_state[S_INDEX(STEER_ANGLE)] =
            max(min(next_state[S_INDEX(STEER_ANGLE)], this->params_.max_steer_angle), -this->params_.max_steer_angle);
        next_state[S_INDEX(STEER_ANGLE_RATE)] = state_der[S_INDEX(STEER_ANGLE)];
        break;
    }
    if (i == S_INDEX(ROLL) || i == S_INDEX(PITCH))
    {
      if (this->tex_helper_->checkTextureUse(0))
      {
        front_left =
            make_float3(front_left.x * cosf(next_state[1]) - front_left.y * sinf(next_state[1]) + next_state[2],
                        front_left.x * sinf(next_state[1]) + front_left.y * cosf(next_state[1]) + next_state[3], 0);
        front_right =
            make_float3(front_right.x * cosf(next_state[1]) - front_right.y * sinf(next_state[1]) + next_state[2],
                        front_right.x * sinf(next_state[1]) + front_right.y * cosf(next_state[1]) + next_state[3], 0);
        rear_left =
            make_float3(rear_left.x * cosf(next_state[1]) - rear_left.y * sinf(next_state[1]) + next_state[2],
                        rear_left.x * sinf(next_state[1]) + rear_left.y * cosf(next_state[1]) + next_state[3], 0);
        rear_right =
            make_float3(rear_right.x * cosf(next_state[1]) - rear_right.y * sinf(next_state[1]) + next_state[2],
                        rear_right.x * sinf(next_state[1]) + rear_right.y * cosf(next_state[1]) + next_state[3], 0);
        front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
        front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
        rear_left_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_left);
        rear_right_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_right);

        // max magnitude
        if (i == S_INDEX(ROLL))
        {
          float front_diff = front_left_height - front_right_height;
          front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
          float rear_diff = rear_left_height - rear_right_height;
          rear_diff = max(min(rear_diff, 0.736 * 2), -0.736 * 2);
          float front_roll = asinf(front_diff / (0.737 * 2));
          float rear_roll = asinf(rear_diff / (0.737 * 2));
          next_state[5] = (front_roll + rear_roll) / 2;
        }
        if (i == S_INDEX(PITCH))
        {
          float left_diff = rear_left_height - front_left_height;
          left_diff = max(min(left_diff, 2.98), -2.98);
          float right_diff = rear_right_height - front_right_height;
          right_diff = max(min(right_diff, 2.98), -2.98);
          float left_pitch = asinf((left_diff) / 2.981);
          float right_pitch = asinf((right_diff) / 2.981);
          next_state[6] = (left_pitch + right_pitch) / 2;
          if (isnan(next_state[6]) || isinf(next_state[6]) || fabsf(next_state[6]) > M_PIf32)
          {
            next_state[6] = 4.0;
          }
        }
        if (isnan(next_state[i]) || isinf(next_state[i]) || fabsf(next_state[i]) > M_PIf32)
        {
          next_state[i] = 4.0;
        }
      }
      else
      {
        next_state[5] = 0;
        next_state[6] = 0;
      }
    }
  }
  __syncthreads();

  float yaw = next_state[S_INDEX(YAW)];
  float q[4];
  mppi::math::Euler2QuatNWU(roll, pitch, yaw, q);
#if false
  for (int i = tdy; i < PARENT_CLASS::OUTPUT_DIM; i += blockDim.y)
  {
    switch (i)
    {
      case O_INDEX(BASELINK_VEL_B_X):
        output[i] = next_state[S_INDEX(VEL_X)];
        break;
      case O_INDEX(BASELINK_POS_I_X):
        output[i] = next_state[S_INDEX(POS_X)];
        break;
      case O_INDEX(BASELINK_POS_I_Y):
        output[i] = next_state[S_INDEX(POS_Y)];
        break;
      case O_INDEX(YAW):
        output[i] = next_state[S_INDEX(YAW)];
        break;
      case O_INDEX(PITCH):
        output[i] = next_state[S_INDEX(PITCH)];
        break;
      case O_INDEX(ROLL):
        output[i] = next_state[S_INDEX(ROLL)];
        break;
      case O_INDEX(ATTITUDE_QW):
        output[i] = q[0];
        break;
      case O_INDEX(ATTITUDE_QX):
        output[i] = q[1];
        break;
      case O_INDEX(ATTITUDE_QY):
        output[i] = q[2];
        break;
      case O_INDEX(ATTITUDE_QZ):
        output[i] = q[3];
        break;
      case O_INDEX(STEER_ANGLE):
        output[i] = next_state[S_INDEX(STEER_ANGLE)];
        break;
      case O_INDEX(STEER_ANGLE_RATE):
        output[i] = next_state[S_INDEX(STEER_ANGLE_RATE)];
        break;
      case O_INDEX(CENTER_POS_I_X):
        output[i] = next_state[S_INDEX(POS_X)];
        break;
      case O_INDEX(CENTER_POS_I_Y):
        output[i] = next_state[S_INDEX(POS_Y)];
        break;
      case O_INDEX(WHEEL_POS_I_FL_X):
        output[i] = front_left.x;
        break;
      case O_INDEX(WHEEL_POS_I_FL_Y):
        output[i] = front_left.y;
        break;
      case O_INDEX(WHEEL_POS_I_FR_X):
        output[i] = front_right.x;
        break;
      case O_INDEX(WHEEL_POS_I_FR_Y):
        output[i] = front_right.y;
        break;
      case O_INDEX(WHEEL_POS_I_RR_X):
        output[i] = rear_right.x;
        break;
      case O_INDEX(WHEEL_POS_I_RR_Y):
        output[i] = rear_right.y;
        break;
      case O_INDEX(WHEEL_POS_I_RL_X):
        output[i] = rear_left.x;
        break;
      case O_INDEX(WHEEL_POS_I_RL_Y):
        output[i] = rear_left.y;
        break;
      case O_INDEX(WHEEL_FORCE_B_FL_Z):
      case O_INDEX(WHEEL_FORCE_B_FR_Z):
      case O_INDEX(WHEEL_FORCE_B_RR_Z):
      case O_INDEX(WHEEL_FORCE_B_RL_Z):
        output[i] = 1000;
        break;
      default:
        output[i] = 0;
    }
  }
#else
  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(BASELINK_POS_I_Z)] = 0;
  output[O_INDEX(OMEGA_B_X)] = 0;
  output[O_INDEX(OMEGA_B_Y)] = 0;
  output[O_INDEX(OMEGA_B_Z)] = 0;
  output[O_INDEX(YAW)] = next_state[S_INDEX(YAW)];
  output[O_INDEX(PITCH)] = next_state[S_INDEX(PITCH)];
  output[O_INDEX(ROLL)] = next_state[S_INDEX(ROLL)];
  output[O_INDEX(ATTITUDE_QW)] = q[0];
  output[O_INDEX(ATTITUDE_QX)] = q[1];
  output[O_INDEX(ATTITUDE_QY)] = q[2];
  output[O_INDEX(ATTITUDE_QZ)] = q[3];
  output[O_INDEX(STEER_ANGLE)] = next_state[S_INDEX(STEER_ANGLE)];
  output[O_INDEX(STEER_ANGLE_RATE)] = next_state[S_INDEX(STEER_ANGLE_RATE)];
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
#endif
}
