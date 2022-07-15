#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>

void RacerDubinsElevation::GPUSetup()
{
  RacerDubinsImpl<RacerDubinsElevation, 9>* derived = static_cast<RacerDubinsImpl<RacerDubinsElevation, 9>*>(this);
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
  RacerDubinsImpl<RacerDubinsElevation, 9>::paramsToDevice();
}

void RacerDubinsElevation::updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt)
{
  state += state_der * dt;
  state(1) = angle_utils::normalizeAngle(state(1));
  state(4) = max(min(state(4), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  state(7) = state_der(4);

  if (this->tex_helper_->checkTextureUse(0))
  {
    float3 front_left = make_float3(2.981, 0.737, 0);
    float3 front_right = make_float3(2.981, -0.737, 0);
    float3 back_left = make_float3(0, 0.737, 0);
    float3 back_right = make_float3(0, -0.737, 0);
    front_left = make_float3(front_left.x * cosf(state(1)) - front_left.y * sinf(state(1)) + state(2),
                             front_left.x * sinf(state(1)) + front_left.y * cosf(state(1)) + state(3), 0);
    front_right = make_float3(front_right.x * cosf(state(1)) - front_right.y * sinf(state(1)) + state(2),
                              front_right.x * sinf(state(1)) + front_right.y * cosf(state(1)) + state(3), 0);
    back_left = make_float3(back_left.x * cosf(state(1)) - back_left.y * sinf(state(1)) + state(2),
                            back_left.x * sinf(state(1)) + back_left.y * cosf(state(1)) + state(3), 0);
    back_right = make_float3(back_right.x * cosf(state(1)) - back_right.y * sinf(state(1)) + state(2),
                             back_right.x * sinf(state(1)) + back_right.y * cosf(state(1)) + state(3), 0);
    float front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
    float front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
    float back_left_height = this->tex_helper_->queryTextureAtWorldPose(0, back_left);
    float back_right_height = this->tex_helper_->queryTextureAtWorldPose(0, back_right);

    float front_diff = front_left_height - front_right_height;
    front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
    float back_diff = back_left_height - back_right_height;
    back_diff = max(min(back_diff, 0.736 * 2), -0.736 * 2);
    float front_roll = asinf(front_diff / (0.737 * 2));
    float back_roll = asinf(back_diff / (0.737 * 2));
    state(5) = (front_roll + back_roll) / 2;

    float left_diff = back_left_height - front_left_height;
    left_diff = max(min(left_diff, 2.98), -2.98);
    float right_diff = back_right_height - front_right_height;
    right_diff = max(min(right_diff, 2.98), -2.98);
    float left_pitch = asinf((left_diff) / 2.981);
    float right_pitch = asinf((right_diff) / 2.981);
    state(6) = (left_pitch + right_pitch) / 2;
  }
  else
  {
    state(5) = 0;
    state(6) = 0;
  }

  if (isnan(state(5)) || isinf(state(5)) || abs(state(5)) > M_PI)
  {
    state(5) = 4.0;
  }
  if (isnan(state(6)) || isinf(state(6)) || abs(state(6)) > M_PI)
  {
    state(6) = 4.0;
  }

  state(8) = state_der(0);

  state_der.setZero();
}

__device__ void RacerDubinsElevation::updateState(float* state, float* state_der, const float dt)
{
  int i;
  int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < 7; i += blockDim.y)
  {
    state[i] += state_der[i] * dt;
    if (i == 0)
    {
      state[8] = state_der[i];
    }
    if (i == 1)
    {
      state[i] = angle_utils::normalizeAngle(state[i]);
    }
    if (i == 4)
    {
      state[4] = max(min(state[4], this->params_.max_steer_angle), -this->params_.max_steer_angle);
      state[7] = state_der[i];
    }
    if (i == 5 || i == 6)
    {
      // roll
      if (this->tex_helper_->checkTextureUse(0))
      {
        float3 front_left = make_float3(2.981, 0.737, 0);
        float3 front_right = make_float3(2.981, -0.737, 0);
        float3 back_left = make_float3(0, 0.737, 0);
        float3 back_right = make_float3(0, -0.737, 0);
        front_left = make_float3(front_left.x * cosf(state[1]) - front_left.y * sinf(state[1]) + state[2],
                                 front_left.x * sinf(state[1]) + front_left.y * cosf(state[1]) + state[3], 0);
        front_right = make_float3(front_right.x * cosf(state[1]) - front_right.y * sinf(state[1]) + state[2],
                                  front_right.x * sinf(state[1]) + front_right.y * cosf(state[1]) + state[3], 0);
        back_left = make_float3(back_left.x * cosf(state[1]) - back_left.y * sinf(state[1]) + state[2],
                                back_left.x * sinf(state[1]) + back_left.y * cosf(state[1]) + state[3], 0);
        back_right = make_float3(back_right.x * cosf(state[1]) - back_right.y * sinf(state[1]) + state[2],
                                 back_right.x * sinf(state[1]) + back_right.y * cosf(state[1]) + state[3], 0);
        float front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
        float front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
        float back_left_height = this->tex_helper_->queryTextureAtWorldPose(0, back_left);
        float back_right_height = this->tex_helper_->queryTextureAtWorldPose(0, back_right);

        // max magnitude
        if (i == 5)
        {
          float front_diff = front_left_height - front_right_height;
          front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
          float back_diff = back_left_height - back_right_height;
          back_diff = max(min(back_diff, 0.736 * 2), -0.736 * 2);
          float front_roll = asinf(front_diff / (0.737 * 2));
          float back_roll = asinf(back_diff / (0.737 * 2));
          if (abs(front_roll) > abs(back_roll))
          {
            state[i] = front_roll;
          }
          else
          {
            state[i] = back_roll;
          }
        }
        if (i == 6)
        {
          float left_diff = back_left_height - front_left_height;
          left_diff = max(min(left_diff, 2.98), -2.98);
          float right_diff = back_right_height - front_right_height;
          right_diff = max(min(right_diff, 2.98), -2.98);
          float left_pitch = asinf((left_diff) / 2.981);
          float right_pitch = asinf((right_diff) / 2.981);
          if (abs(left_pitch) > abs(right_pitch))
          {
            state[i] = left_pitch;
          }
          else
          {
            state[i] = right_pitch;
          }
        }

        if (isnan(state[i]) || isinf(state[i]) || abs(state[i]) > M_PI)
        {
          // printf("got invalid roll %f from %f %f diff %f %f\n", state[i], front_left_height, front_right_height,
          // diff, (diff) / (0.737 * 2)); printf("got invalid roll at points (%f %f) (%f, %f)\n", front_left.x,
          // front_left.y, front_right.x, front_right.y);
          state[i] = 4.0;
        }
      }
      else
      {
        state[i] = 0;
      }
    }
    state_der[i] = 0;  // Important: reset the state derivative to zero.
  }
}

void RacerDubinsElevation::computeDynamics(const Eigen::Ref<const state_array>& state,
                                           const Eigen::Ref<const control_array>& control,
                                           Eigen::Ref<state_array> state_der)
{
  bool enable_brake = control(0) < 0;
  // applying position throttle
  state_der(0) = (!enable_brake) * this->params_.c_t * control(0) +
                 (enable_brake) * this->params_.c_b * control(0) * (state(0) >= 0 ? 1 : -1) -
                 this->params_.c_v * state(0) + this->params_.c_0;
  if (abs(state[6]) < M_PI)
  {
    state_der[0] -= this->params_.gravity * sinf(state[6]);
  }
  state_der(1) = (state(0) / this->params_.wheel_base) * tan(state(4));
  state_der(2) = state(0) * cosf(state(1));
  state_der(3) = state(0) * sinf(state(1));
  state_der(4) = (control(1) / this->params_.steer_command_angle_scale - state(4)) * this->params_.steering_constant;
  state_der(4) = max(min(state_der(4), this->params_.max_steer_rate), -this->params_.max_steer_rate);
}

__device__ void RacerDubinsElevation::computeDynamics(float* state, float* control, float* state_der, float* theta_s)
{
  bool enable_brake = control[0] < 0;
  // applying position throttle
  state_der[0] = (!enable_brake) * this->params_.c_t * control[0] +
                 (enable_brake) * this->params_.c_b * control[0] * (state[0] >= 0 ? 1 : -1) -
                 this->params_.c_v * state[0] + this->params_.c_0;
  if (abs(state[6]) < M_PI)
  {
    state_der[0] -= this->params_.gravity * sinf(state[6]);
  }
  state_der[1] = (state[0] / this->params_.wheel_base) * tan(state[4]);
  state_der[2] = state[0] * cosf(state[1]);
  state_der[3] = state[0] * sinf(state[1]);
  state_der[4] = (control[1] / this->params_.steer_command_angle_scale - state[4]) * this->params_.steering_constant;
  state_der[4] = max(min(state_der[4], this->params_.max_steer_rate), -this->params_.max_steer_rate);
}
