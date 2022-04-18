#include <mppi/dynamics/racer_suspension/racer_suspension.cuh>

void RacerSuspension::GPUSetup()
{
  auto* derived = static_cast<Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>*>(this);
  tex_helper_->GPUSetup();
  derived->GPUSetup();
}

void RacerSuspension::freeCudaMem()
{
  tex_helper_->freeCudaMem();
}

void RacerSuspension::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    // does all the internal texture updates
    tex_helper_->copyToDevice();
    // makes sure that the device ptr sees the correct texture object
    HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->tex_helper_), &(tex_helper_->ptr_d_),
                                 sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
  }
  Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>::paramsToDevice();
}

void RacerSuspension::updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt)
{
  state += state_der * dt;
  state(1) = angle_utils::normalizeAngle(state(1));
  state(4) -= state_der(4) * dt;
  state(4) = state_der(4) + (state(4) - state_der(4)) * expf(-this->params_.steering_constant * dt);
  state_der.setZero();
}

__device__ void RacerSuspension::updateState(float* state, float* state_der, const float dt)
{
  unsigned int i;
  unsigned int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < STATE_DIM; i += blockDim.y)
  {
    state[i] += state_der[i] * dt;
  }
}

void RacerSuspension::computeDynamics(const Eigen::Ref<const state_array>& state,
                                      const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der)
{
  Dynamics::computeDynamics(state, control, state_der);
}

__device__ void RacerSuspension::computeDynamics(float* state, float* control, float* state_der, float* theta_s)
{
  bool enable_brake = control[0] < 0;
  // applying position throttle
  state_der[0] = (!enable_brake) * this->params_.c_t * control[0] +
                 (enable_brake) * this->params_.c_b * control[0] * state[0] - this->params_.c_v * state[0] +
                 this->params_.c_0;
  if (abs(state[6]) < M_PI)
  {
    state_der[0] -= this->params_.gravity * sinf(state[6]);
  }
  state_der[1] = (state[0] / this->params_.wheel_base) * tan(state[4]);
  state_der[2] = state[0] * cosf(state[1]);
  state_der[3] = state[0] * sinf(state[1]);
  state_der[4] = control[1] / this->params_.steer_command_angle_scale;
}
