//
// Created by jason on 8/31/22.
//

#include "racer_dubins_elevation_lstm_steering.cuh"

RacerDubinsElevationLSTMSteering::RacerDubinsElevationLSTMSteering(cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering, RacerDubinsElevationParams>(stream)
{
  this->requires_buffer_ = true;
  lstm_lstm_helper_ = std::make_shared<NN>(stream);
}

RacerDubinsElevationLSTMSteering::RacerDubinsElevationLSTMSteering(RacerDubinsElevationParams& params,
                                                                   cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering, RacerDubinsElevationParams>(params, stream)
{
  this->requires_buffer_ = true;
  lstm_lstm_helper_ = std::make_shared<NN>(stream);
}

RacerDubinsElevationLSTMSteering::RacerDubinsElevationLSTMSteering(std::string path, cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering, RacerDubinsElevationParams>(stream)
{
  if (!fileExists(path))
  {
    std::cerr << "Could not load neural net model at path: " << path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(path);
  this->params_.max_steer_rate = param_dict.at("parameters/max_rate_pos").data<float>()[0];
  this->params_.steering_constant = param_dict.at("parameters/constant").data<float>()[0];
  this->params_.steer_accel_constant = param_dict.at("parameters/accel_constant").data<float>()[0];
  this->params_.steer_accel_drag_constant = param_dict.at("parameters/accel_drag_constant").data<float>()[0];
  lstm_lstm_helper_ = std::make_shared<NN>(path, stream);
  this->requires_buffer_ = true;
}

void RacerDubinsElevationLSTMSteering::GPUSetup()
{
  lstm_lstm_helper_->GPUSetup();
  // makes sure that the device ptr sees the correct lstm model
  this->network_d_ = lstm_lstm_helper_->getLSTMDevicePtr();
  PARENT_CLASS::GPUSetup();
}

void RacerDubinsElevationLSTMSteering::freeCudaMem()
{
  PARENT_CLASS::freeCudaMem();
  lstm_lstm_helper_->freeCudaMem();
}

void RacerDubinsElevationLSTMSteering::step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
                                            Eigen::Ref<state_array> state_der,
                                            const Eigen::Ref<const control_array>& control,
                                            Eigen::Ref<output_array> output, const float t, const float dt)
{
  this->computeParametricDelayDeriv(state, control, state_der);
  this->computeParametricAccelDeriv(state, control, state_der, dt);

  const float parametric_accel = fmaxf(
      fminf((control(C_INDEX(STEER_CMD)) * this->params_.steer_command_angle_scale - state(S_INDEX(STEER_ANGLE))) *
                this->params_.steering_constant,
            this->params_.max_steer_rate),
      -this->params_.max_steer_rate);
  state_der(S_INDEX(STEER_ANGLE_RATE)) =
      (parametric_accel - state(S_INDEX(STEER_ANGLE_RATE))) * this->params_.steer_accel_constant -
      state(S_INDEX(STEER_ANGLE_RATE)) * this->params_.steer_accel_drag_constant;

  LSTM::input_array input;
  input(0) = state(S_INDEX(STEER_ANGLE)) * 0.2f;
  input(1) = state(S_INDEX(STEER_ANGLE_RATE)) * 0.2f;
  input(2) = control(C_INDEX(STEER_CMD));
  input(3) = state_der(S_INDEX(STEER_ANGLE_RATE)) * 0.2f;  // this is the parametric part as input
  LSTM::output_array nn_output = LSTM::output_array::Zero();
  lstm_lstm_helper_->forward(input, nn_output);
  state_der(S_INDEX(STEER_ANGLE_RATE)) += nn_output(0) * 5.0f;
  state_der(S_INDEX(STEER_ANGLE)) = state(S_INDEX(STEER_ANGLE_RATE));

  // Integrate using racer_dubins updateState
  updateState(state, next_state, state_der, dt);
  SharedBlock sb;
  computeUncertaintyPropagation(state.data(), control.data(), state_der.data(), next_state.data(), dt, &this->params_,
                                &sb);

  float roll = state(S_INDEX(ROLL));
  float pitch = state(S_INDEX(PITCH));
  RACER::computeStaticSettling<DYN_PARAMS_T::OutputIndex, TwoDTextureHelper<float>>(
      this->tex_helper_, next_state(S_INDEX(YAW)), next_state(S_INDEX(POS_X)), next_state(S_INDEX(POS_Y)), roll, pitch,
      output.data());
  next_state[S_INDEX(PITCH)] = pitch;
  next_state[S_INDEX(ROLL)] = roll;

  setOutputs(state_der.data(), next_state.data(), output.data());
}

__device__ void RacerDubinsElevationLSTMSteering::initializeDynamics(float* state, float* control, float* output,
                                                                     float* theta_s, float t_0, float dt)
{
  const int shift = (mppi::math::int_multiple_const(PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
                     blockDim.x * blockDim.z *
                         mppi::math::int_multiple_const(PARENT_CLASS::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4))) /
                    sizeof(float);
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* shared_params = (DYN_PARAMS_T*)theta_s;
    *shared_params = this->params_;
  }
  network_d_->initialize(theta_s + shift);
  setOutputs(state, state, output);
}

__device__ inline void RacerDubinsElevationLSTMSteering::step(float* state, float* next_state, float* state_der,
                                                              float* control, float* output, float* theta_s,
                                                              const float t, const float dt)
{
  DYN_PARAMS_T* params_p;
  SharedBlock *sb_mem, *sb;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta_s;
  }
  else
  {
    params_p = &(this->params_);
  }
  if (PARENT_CLASS::SHARED_MEM_REQUEST_BLK_BYTES != 0)
  {
    sb_mem = (SharedBlock*)&theta_s[mppi::math::int_multiple_const(PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES,
                                                                   sizeof(float4)) /
                                    sizeof(float)];
    sb = &sb_mem[threadIdx.x + blockDim.x * threadIdx.z];
  }
  computeParametricDelayDeriv(state, control, state_der, params_p);
  computeParametricAccelDeriv(state, control, state_der, dt, params_p);

  // computes the velocity dot

  const uint tdy = threadIdx.y;

  const int shift = (mppi::math::int_multiple_const(PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES, sizeof(float4)) +
                     blockDim.x * blockDim.z *
                         mppi::math::int_multiple_const(PARENT_CLASS::SHARED_MEM_REQUEST_BLK_BYTES, sizeof(float4))) /
                    sizeof(float);
  // loads in the input to the network
  float* input_loc = network_d_->getInputLocation(theta_s + shift);
  if (tdy == 0)
  {
    const float parametric_accel =
        fmaxf(fminf((control[C_INDEX(STEER_CMD)] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
                        params_p->steering_constant,
                    params_p->max_steer_rate),
              -params_p->max_steer_rate);
    state_der[S_INDEX(STEER_ANGLE_RATE)] =
        (parametric_accel - state[S_INDEX(STEER_ANGLE_RATE)]) * params_p->steer_accel_constant -
        state[S_INDEX(STEER_ANGLE_RATE)] * params_p->steer_accel_drag_constant;

    input_loc[0] = state[S_INDEX(STEER_ANGLE)] * 0.2f;
    input_loc[1] = state[S_INDEX(STEER_ANGLE_RATE)] * 0.2f;
    input_loc[2] = control[C_INDEX(STEER_CMD)];
    input_loc[3] = state_der[S_INDEX(STEER_ANGLE_RATE)] * 0.2f;  // this is the parametric part as input
  }
  __syncthreads();
  // runs the network
  float* nn_output = network_d_->forward(nullptr, theta_s + shift);
  // copies the results of the network to state derivative
  if (tdy == 0)
  {
    state_der[S_INDEX(STEER_ANGLE_RATE)] += nn_output[0] * 5.0f;
    state_der[S_INDEX(STEER_ANGLE)] = state[S_INDEX(STEER_ANGLE_RATE)];
  }
  __syncthreads();

  updateState(state, next_state, state_der, dt);
  computeUncertaintyPropagation(state, control, state_der, next_state, dt, params_p, sb);
  if (tdy == 0)
  {
    float roll = state[S_INDEX(ROLL)];
    float pitch = state[S_INDEX(PITCH)];
    RACER::computeStaticSettling<DYN_PARAMS_T::OutputIndex, TwoDTextureHelper<float>>(
        this->tex_helper_, next_state[S_INDEX(YAW)], next_state[S_INDEX(POS_X)], next_state[S_INDEX(POS_Y)], roll,
        pitch, output);
    next_state[S_INDEX(PITCH)] = pitch;
    next_state[S_INDEX(ROLL)] = roll;
  }
  __syncthreads();
  setOutputs(state_der, next_state, output);
}

void RacerDubinsElevationLSTMSteering::updateFromBuffer(const buffer_trajectory& buffer)
{
  NN::init_buffer init_buffer;
  if (buffer.find("STEER_ANGLE") == buffer.end() || buffer.find("STEER_ANGLE_RATE") == buffer.end() ||
      buffer.find("STEER_CMD") == buffer.end())
  {
    std::cout << "WARNING: not using init buffer" << std::endl;
    for (const auto& it : buffer)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return;
  }

  init_buffer.row(0) = buffer.at("STEER_ANGLE");
  init_buffer.row(1) = buffer.at("STEER_ANGLE_RATE");
  init_buffer.row(2) = buffer.at("STEER_CMD");

  lstm_lstm_helper_->initializeLSTM(init_buffer);
}

void RacerDubinsElevationLSTMSteering::initializeDynamics(const Eigen::Ref<const state_array>& state,
                                                          const Eigen::Ref<const control_array>& control,
                                                          Eigen::Ref<output_array> output, float t_0, float dt)
{
  this->lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  PARENT_CLASS::initializeDynamics(state, control, output, t_0, dt);
}

__device__ void RacerDubinsElevationLSTMSteering::updateState(float* state, float* next_state, float* state_der,
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
      next_state[i] = fmaxf(fminf(next_state[i], this->params_.max_steer_angle), -this->params_.max_steer_angle);
      next_state[S_INDEX(STEER_ANGLE_RATE)] =
          state[S_INDEX(STEER_ANGLE_RATE)] + state_der[S_INDEX(STEER_ANGLE_RATE)] * dt;
    }
    if (i == S_INDEX(BRAKE_STATE))
    {
      next_state[i] = fminf(fmaxf(next_state[i], 0.0f), 1.0f);
    }
  }
}

void RacerDubinsElevationLSTMSteering::updateState(const Eigen::Ref<const state_array> state,
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
      fmaxf(fminf(next_state(S_INDEX(STEER_ANGLE)), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  next_state(S_INDEX(STEER_ANGLE_RATE)) = state(S_INDEX(STEER_ANGLE_RATE)) + state_der(S_INDEX(STEER_ANGLE_RATE)) * dt;
  next_state(S_INDEX(BRAKE_STATE)) =
      fminf(fmaxf(next_state(S_INDEX(BRAKE_STATE)), 0.0f), -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x);
}
