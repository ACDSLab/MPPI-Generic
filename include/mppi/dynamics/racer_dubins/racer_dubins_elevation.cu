#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>

void RacerDubinsElevation::GPUSetup()
{
  RacerDubinsImpl<RacerDubinsElevation, 7>* derived = static_cast<RacerDubinsImpl<RacerDubinsElevation, 7>*>(this);
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
  RacerDubinsImpl<RacerDubinsElevation, 7>::paramsToDevice();
}

void RacerDubinsElevation::updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt)
{
  state += state_der * dt;
  state(1) = angle_utils::normalizeAngle(state(1));
  state(4) -= state_der(4) * dt;
  state(4) = state_der(4) + (state(4) - state_der(4)) * expf(-this->params_.steering_constant * dt);
  state_der.setZero();
}

__device__ void RacerDubinsElevation::updateState(float* state, float* state_der, const float dt)
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
    }
    if (i == 5)
    {
      // roll
      if (this->tex_helper_->checkTextureUse(0))
      {
        float3 front_left = make_float3(2.981, 0.737, 0);
        float3 front_right = make_float3(2.981, -0.737, 0);
        front_left = make_float3(front_left.x * cosf(state[1]) - front_left.y * sinf(state[1]) + state[2],
                                 front_left.x * sinf(state[1]) + front_left.y * cosf(state[1]) + state[3], 0);
        front_right = make_float3(front_right.x * cosf(state[1]) - front_right.y * sinf(state[1]) + state[2],
                                  front_right.x * sinf(state[1]) + front_right.y * cosf(state[1]) + state[3], 0);
        float front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
        float front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
        state[i] = asinf((front_left_height - front_right_height) / (0.737 * 2));
      }
      else
      {
        state[i] = 0;
      }
      if (isnan(state[i]) || isinf(state[i]) || abs(state[i]) > M_PI)
      {
        state[i] = 4.0;
      }
    }
    if (i == 6)
    {
      // pitch
      if (this->tex_helper_->checkTextureUse(0))
      {
        float3 front_left = make_float3(2.981, 0.737, 0);
        float3 back_left = make_float3(0, 0.737, 0);
        front_left = make_float3(front_left.x * cosf(state[1]) - front_left.y * sinf(state[1]),
                                 front_left.x * sinf(state[1]) + front_left.y * cosf(state[1]), 0);
        back_left = make_float3(back_left.x * cosf(state[1]) - back_left.y * sinf(state[1]),
                                back_left.x * sinf(state[1]) + back_left.y * cosf(state[1]), 0);
        float front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
        float back_left_height = this->tex_helper_->queryTextureAtWorldPose(0, back_left);
        state[i] = asinf((back_left_height - front_left_height) / 2.981);
      }
      else
      {
        state[i] = 0;
      }
      if (isnan(state[i]) || isinf(state[i]) || abs(state[i]) > M_PI)
      {
        state[i] = 4.0;
      }
    }
    state_der[i] = 0;  // Important: reset the state derivative to zero.
  }
}
