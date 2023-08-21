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
  : RacerDubinsElevationLSTMSteering(stream)
{
  if (!fileExists(path))
  {
    std::cerr << "Could not load neural net model at path: " << path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(path);
  this->params_.max_steer_rate = param_dict.at("params/max_steer_rate").data<float>()[0];
  this->params_.steering_constant = param_dict.at("params/steering_constant").data<float>()[0];
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
  this->computeParametricSteerDeriv(state, control, state_der);
  this->computeParametricAccelDeriv(state, control, state_der, dt);

  LSTM::input_array input;
  input(0) = state(S_INDEX(VEL_X));
  input(1) = state(S_INDEX(STEER_ANGLE));
  input(2) = state(S_INDEX(STEER_ANGLE_RATE));
  input(3) = control(C_INDEX(STEER_CMD));
  input(4) = state_der(S_INDEX(STEER_ANGLE));  // this is the parametric part as input
  LSTM::output_array nn_output = LSTM::output_array::Zero();
  lstm_lstm_helper_->forward(input, nn_output);
  state_der(S_INDEX(STEER_ANGLE)) += nn_output(0) * 10.0f;

  // Integrate using racer_dubins updateState
  this->PARENT_CLASS::updateState(state, next_state, state_der, dt);

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
  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* shared_params = (DYN_PARAMS_T*)theta_s;
    *shared_params = this->params_;
  }
  network_d_->initialize(theta_s + shift);
  for (int i = 0; i < OUTPUT_DIM && i < STATE_DIM; i++)
  {
    output[i] = state[i];
  }
}

__device__ inline void RacerDubinsElevationLSTMSteering::step(float* state, float* next_state, float* state_der,
                                                              float* control, float* output, float* theta_s,
                                                              const float t, const float dt)
{
  DYN_PARAMS_T* params_p;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
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
  const uint tdy = threadIdx.y;

  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES / 4 + 1;
  // loads in the input to the network
  float* input_loc = network_d_->getInputLocation(theta_s + shift);
  if (tdy == 0)
  {
    input_loc[0] = state[S_INDEX(VEL_X)];
    input_loc[1] = state[S_INDEX(STEER_ANGLE)];
    input_loc[2] = state[S_INDEX(STEER_ANGLE_RATE)];
    input_loc[3] = control[C_INDEX(STEER_CMD)];
    input_loc[4] = state_der[S_INDEX(STEER_ANGLE)];  // this is the parametric part as input
  }
  __syncthreads();
  // runs the network
  float* nn_output = network_d_->forward(nullptr, theta_s + shift);
  // copies the results of the network to state derivative
  if (threadIdx.y == 0)
  {
    state_der[S_INDEX(STEER_ANGLE)] += nn_output[0] * 10.0f;
  }
  __syncthreads();

  updateState(state, next_state, state_der, dt, params_p);
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
  setOutputs(state_der, next_state, output);
}

void RacerDubinsElevationLSTMSteering::updateFromBuffer(const buffer_trajectory& buffer)
{
  NN::init_buffer init_buffer;
  if (buffer.find("VEL_X") == buffer.end() || buffer.find("STEER_ANGLE") == buffer.end() ||
      buffer.find("STEER_ANGLE_RATE") == buffer.end() || buffer.find("STEER_CMD") == buffer.end())
  {
    std::cout << "WARNING: not using init buffer" << std::endl;
    for (const auto& it : buffer)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return;
  }

  init_buffer.row(0) = buffer.at("VEL_X");
  init_buffer.row(1) = buffer.at("STEER_ANGLE");
  init_buffer.row(2) = buffer.at("STEER_ANGLE_RATE");
  init_buffer.row(3) = buffer.at("STEER_CMD");

  lstm_lstm_helper_->initializeLSTM(init_buffer);
}

void RacerDubinsElevationLSTMSteering::initializeDynamics(const Eigen::Ref<const state_array>& state,
                                                          const Eigen::Ref<const control_array>& control,
                                                          Eigen::Ref<output_array> output, float t_0, float dt)
{
  this->lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  PARENT_CLASS::initializeDynamics(state, control, output, t_0, dt);
}
