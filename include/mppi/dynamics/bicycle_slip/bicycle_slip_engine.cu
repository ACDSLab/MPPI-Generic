//
// Created by jason on 9/7/22.
//

#include "bicycle_slip_engine.cuh"

BicycleSlipEngine::BicycleSlipEngine(cudaStream_t stream)
  : MPPI_internal::Dynamics<BicycleSlipEngine, BicycleSlipEngineParams>(stream)
{
  this->requires_buffer_ = true;
  tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  steer_lstm_lstm_helper_ = std::make_shared<STEER_NN>(stream);
  delay_lstm_lstm_helper_ = std::make_shared<DELAY_NN>(stream);
  engine_lstm_lstm_helper_ = std::make_shared<ENGINE_NN>(stream);
  terra_lstm_lstm_helper_ = std::make_shared<TERRA_NN>(stream);
}

BicycleSlipEngine::BicycleSlipEngine(std::string ackerman_path, cudaStream_t stream)
  : MPPI_internal::Dynamics<BicycleSlipEngine, BicycleSlipEngineParams>(stream)
{
  this->requires_buffer_ = true;
  tex_helper_ = new TwoDTextureHelper<float>(1, stream);

  delay_lstm_lstm_helper_ = std::make_shared<DELAY_NN>(stream);
  engine_lstm_lstm_helper_ = std::make_shared<ENGINE_NN>(stream);
  terra_lstm_lstm_helper_ = std::make_shared<TERRA_NN>(stream);
  steer_lstm_lstm_helper_ = std::make_shared<STEER_NN>(stream);

  if (!fileExists(ackerman_path))
  {
    std::cerr << "Could not load neural net model at ackerman_path: " << ackerman_path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(ackerman_path);
  this->params_.gravity = param_dict.at("params/gravity").data<float>()[0];
  this->params_.wheel_angle_scale = param_dict.at("params/wheel_angle_scale").data<float>()[0];
  this->params_.steer_angle_scale = param_dict.at("params/steer_angle_scale").data<float>()[0];

  // load the delay params
  this->params_.brake_delay_constant = param_dict.at("delay_model/params/brake_constant").data<float>()[0];
  this->params_.max_brake_rate_neg = param_dict.at("delay_model/params/max_brake_rate_neg").data<float>()[0];
  this->params_.max_brake_rate_pos = param_dict.at("delay_model/params/max_brake_rate_pos").data<float>()[0];

  // load the steering parameters
  this->params_.max_steer_rate = param_dict.at("steer_model/params/max_steer_rate").data<float>()[0];
  this->params_.steering_constant = param_dict.at("steer_model/params/steering_constant").data<float>()[0];

  delay_lstm_lstm_helper_->loadParams("delay_model/model", ackerman_path);
  terra_lstm_lstm_helper_->loadParams("terra_model", ackerman_path);
  engine_lstm_lstm_helper_->loadParams("engine_model", ackerman_path);
  steer_lstm_lstm_helper_->loadParams("steer_model/model", ackerman_path);
}

void BicycleSlipEngine::initializeDynamics(const Eigen::Ref<const state_array>& state,
                                           const Eigen::Ref<const control_array>& control,
                                           Eigen::Ref<output_array> output, float t_0, float dt)
{
  this->steer_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->delay_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->engine_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->terra_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  PARENT_CLASS::initializeDynamics(state, control, output, t_0, dt);
}

MPPI_internal::Dynamics<BicycleSlipEngine, BicycleSlipEngineParams>::state_array
BicycleSlipEngine::stateFromMap(const std::map<std::string, float>& map)
{
  state_array s = state_array::Zero();
  if (map.find("VEL_X") == map.end() || map.find("VEL_Y") == map.end() || map.find("POS_X") == map.end() ||
      map.find("POS_Y") == map.end() || map.find("ROLL") == map.end() || map.find("PITCH") == map.end())
  {
    std::cout << "WARNING: could not find all keys for ackerman slip dynamics" << std::endl;
    for (const auto& it : map)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return s;
  }
  s(S_INDEX(POS_X)) = map.at("POS_X");
  s(S_INDEX(POS_Y)) = map.at("POS_Y");
  s(S_INDEX(VEL_X)) = map.at("VEL_X");
  s(S_INDEX(VEL_Y)) = map.at("VEL_Y");
  s(S_INDEX(OMEGA_Z)) = map.at("OMEGA_Z");
  s(S_INDEX(YAW)) = map.at("YAW");
  s(S_INDEX(ROLL)) = map.at("ROLL");
  s(S_INDEX(PITCH)) = map.at("PITCH");
  if (map.find("STEER_ANGLE") != map.end())
  {
    s(S_INDEX(STEER_ANGLE)) = map.at("STEER_ANGLE");
    s(S_INDEX(STEER_ANGLE_RATE)) = map.at("STEER_ANGLE_RATE");
  }
  else
  {
    std::cout << "WARNING: unable to find BRAKE_STATE or STEER_ANGLE_RATE, using 0" << std::endl;
    s(S_INDEX(STEER_ANGLE)) = 0;
    s(S_INDEX(STEER_ANGLE_RATE)) = 0;
  }
  if (map.find("BRAKE_STATE") != map.end())
  {
    s(S_INDEX(BRAKE_STATE)) = map.at("BRAKE_STATE");
  }
  else if (map.find("BRAKE_CMD") != map.end())
  {
    std::cout << "WARNING: unable to find BRAKE_STATE" << std::endl;
    s(S_INDEX(BRAKE_STATE)) = map.at("BRAKE_CMD");
  }
  else
  {
    std::cout << "WARNING: unable to find BRAKE_CMD or BRAKE_STATE" << std::endl;
    s(S_INDEX(BRAKE_STATE)) = 0;
  }
  return s;
}

void BicycleSlipEngine::updateFromBuffer(const buffer_trajectory& buffer)
{
  if (buffer.find("VEL_X") == buffer.end() || buffer.find("VEL_Y") == buffer.end() ||
      buffer.find("STEER_ANGLE") == buffer.end() || buffer.find("STEER_ANGLE_RATE") == buffer.end() ||
      buffer.find("STEER_CMD") == buffer.end() || buffer.find("BRAKE_STATE") == buffer.end())
  {
    std::cout << "WARNING: not using init buffer" << std::endl;
    for (const auto& it : buffer)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return;
  }

  STEER_NN::init_buffer steer_init_buffer;
  steer_init_buffer.row(0) = buffer.at("VEL_X");
  steer_init_buffer.row(1) = buffer.at("STEER_ANGLE");
  steer_init_buffer.row(2) = buffer.at("STEER_ANGLE_RATE");
  steer_init_buffer.row(3) = buffer.at("STEER_CMD");
  steer_lstm_lstm_helper_->initializeLSTM(steer_init_buffer);

  DELAY_NN::init_buffer delay_init_buffer;
  delay_init_buffer.row(0) = buffer.at("BRAKE_STATE");
  delay_init_buffer.row(1) = buffer.at("BRAKE_CMD");
  delay_lstm_lstm_helper_->initializeLSTM(delay_init_buffer);

  ENGINE_NN::init_buffer engine_init_buffer;
  engine_init_buffer.row(0) = buffer.at("VEL_X");
  engine_init_buffer.row(1) = buffer.at("THROTTLE_CMD");
  engine_init_buffer.row(2) = buffer.at("BRAKE_STATE");
  engine_lstm_lstm_helper_->initializeLSTM(engine_init_buffer);

  TERRA_NN::init_buffer terra_init_buffer;
  terra_init_buffer.row(0) = buffer.at("VEL_X");
  terra_init_buffer.row(1) = buffer.at("VEL_Y");
  terra_init_buffer.row(2) = buffer.at("OMEGA_Z");
  terra_init_buffer.row(3) = buffer.at("STEER_ANGLE");
  terra_init_buffer.row(4) = buffer.at("STEER_ANGLE_RATE");
  // TODO should be pulled from elevation map to be entirely correct
  terra_init_buffer.row(5) = buffer.at("PITCH").unaryExpr([](float x) { return sinf(x); }) * this->params_.gravity;
  terra_init_buffer.row(6) = buffer.at("ROLL").unaryExpr([](float x) { return sinf(x); }) * this->params_.gravity;
  terra_init_buffer.row(7) =
      buffer.at("STEER_ANGLE").unaryExpr([this](float x) { return tanf(x / this->params_.wheel_angle_scale); });
  terra_lstm_lstm_helper_->initializeLSTM(terra_init_buffer);
}

void BicycleSlipEngine::GPUSetup()
{
  steer_lstm_lstm_helper_->GPUSetup();
  delay_lstm_lstm_helper_->GPUSetup();
  engine_lstm_lstm_helper_->GPUSetup();
  terra_lstm_lstm_helper_->GPUSetup();

  // makes sure that the device ptr sees the correct lstm model
  this->steer_network_d_ = steer_lstm_lstm_helper_->getLSTMDevicePtr();
  this->delay_network_d_ = delay_lstm_lstm_helper_->getLSTMDevicePtr();
  this->engine_network_d_ = engine_lstm_lstm_helper_->getLSTMDevicePtr();
  this->terra_network_d_ = terra_lstm_lstm_helper_->getLSTMDevicePtr();

  PARENT_CLASS::GPUSetup();
  tex_helper_->GPUSetup();
  // makes sure that the device ptr sees the correct texture object
  HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->tex_helper_), &(tex_helper_->ptr_d_),
                               sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
}

void BicycleSlipEngine::freeCudaMem()
{
  steer_lstm_lstm_helper_->freeCudaMem();
  delay_lstm_lstm_helper_->freeCudaMem();
  engine_lstm_lstm_helper_->freeCudaMem();
  terra_lstm_lstm_helper_->freeCudaMem();
  tex_helper_->freeCudaMem();
  Dynamics::freeCudaMem();
}

void BicycleSlipEngine::paramsToDevice()
{
  // does all the internal texture updates
  tex_helper_->copyToDevice();
  PARENT_CLASS::paramsToDevice();
}

void BicycleSlipEngine::computeDynamics(const Eigen::Ref<const state_array>& state,
                                        const Eigen::Ref<const control_array>& control,
                                        Eigen::Ref<state_array> state_der)
{
  state_der = state_array::Zero();
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  float brake_cmd = -enable_brake * control(C_INDEX(THROTTLE_BRAKE));
  float throttle_cmd = !enable_brake * control(C_INDEX(THROTTLE_BRAKE));

  state_der(S_INDEX(BRAKE_STATE)) =
      min(max((brake_cmd - state(S_INDEX(BRAKE_STATE))) * this->params_.brake_delay_constant,
              -this->params_.max_brake_rate_neg),
          this->params_.max_brake_rate_pos);
  // TODO if low speed allow infinite brake, not sure if needed
  // TODO need parametric reverse

  // kinematics component
  state_der(S_INDEX(POS_X)) =
      state(S_INDEX(VEL_X)) * cosf(state(S_INDEX(YAW))) - state(S_INDEX(VEL_Y)) * sinf(state(S_INDEX(YAW)));
  state_der(S_INDEX(POS_Y)) =
      state(S_INDEX(VEL_X)) * sinf(state(S_INDEX(YAW))) + state(S_INDEX(VEL_Y)) * cosf(state(S_INDEX(YAW)));

  // runs the brake model
  DELAY_LSTM::input_array brake_input;
  brake_input(0) = state(S_INDEX(BRAKE_STATE));
  brake_input(1) = brake_cmd;
  brake_input(2) = state_der(S_INDEX(BRAKE_STATE));  // stand in for y velocity
  DELAY_LSTM::output_array brake_output = DELAY_LSTM::output_array::Zero();
  delay_lstm_lstm_helper_->forward(brake_input, brake_output);
  state_der(S_INDEX(BRAKE_STATE)) += brake_output(0);

  // runs the engine model
  ENGINE_LSTM::input_array engine_input;
  engine_input(0) = throttle_cmd;
  engine_input(1) = state(S_INDEX(VEL_X));
  engine_input(2) = state(S_INDEX(BRAKE_STATE));
  ENGINE_LSTM::output_array engine_output_arr = ENGINE_LSTM::output_array::Zero();
  engine_lstm_lstm_helper_->forward(engine_input, engine_output_arr);
  const float engine_output = engine_output_arr(0) * 10;

  // runs the parametric part of the steering model
  state_der(S_INDEX(STEER_ANGLE)) =
      (control(C_INDEX(STEER_CMD)) * this->params_.steer_command_angle_scale - state(S_INDEX(STEER_ANGLE))) *
      this->params_.steering_constant;
  state_der(S_INDEX(STEER_ANGLE)) =
      max(min(state_der(S_INDEX(STEER_ANGLE)), this->params_.max_steer_rate), -this->params_.max_steer_rate);

  // runs the steering model
  STEER_LSTM::input_array steer_input;
  steer_input(0) = state(S_INDEX(VEL_X));
  steer_input(1) = state(S_INDEX(STEER_ANGLE));
  steer_input(2) = state(S_INDEX(STEER_ANGLE_RATE));
  steer_input(3) = control(C_INDEX(STEER_CMD));
  steer_input(4) = state_der(S_INDEX(STEER_ANGLE));  // this is the parametric part as input
  STEER_LSTM::output_array steer_output = STEER_LSTM::output_array::Zero();
  steer_lstm_lstm_helper_->forward(steer_input, steer_output);
  state_der(S_INDEX(STEER_ANGLE)) += steer_output(0) * 10;

  const float delta = tanf(state(S_INDEX(STEER_ANGLE)) / this->params_.wheel_angle_scale);
  const float param_yaw_rate = (state(S_INDEX(VEL_X)) / this->params_.wheel_base) *
                               tan(state(S_INDEX(STEER_ANGLE)) / this->params_.steer_angle_scale);

  // runs the terra dynamics model
  TERRA_LSTM::input_array terra_input;
  terra_input(0) = state(S_INDEX(VEL_X));
  terra_input(1) = state(S_INDEX(VEL_Y));
  terra_input(2) = state(S_INDEX(OMEGA_Z));
  terra_input(3) = state(S_INDEX(STEER_ANGLE));
  terra_input(4) = state(S_INDEX(STEER_ANGLE_RATE));
  terra_input(5) = sinf(state(S_INDEX(PITCH))) * this->params_.gravity;
  terra_input(6) = sinf(state(S_INDEX(ROLL))) * this->params_.gravity;
  terra_input(7) = engine_output;
  terra_input(8) = delta;
  terra_input(9) = param_yaw_rate;
  TERRA_LSTM::output_array terra_output = TERRA_LSTM::output_array::Zero();
  terra_lstm_lstm_helper_->forward(terra_input, terra_output);

  const float c_delta = cosf(delta);
  const float s_delta = sinf(delta);
  const float drag_x = terra_output(0) * 10.0f;
  const float drag_y = terra_output(1) * 10.0f;
  const float drag_yaw = terra_output(2) * 10.0f;

  // combine to compute state derivative
  state_der(S_INDEX(VEL_X)) = c_delta * engine_output + engine_output - drag_x * c_delta + drag_y * s_delta - drag_x;
  state_der(S_INDEX(VEL_Y)) = s_delta * engine_output - drag_x * s_delta - drag_y * c_delta - drag_y;
  state_der(S_INDEX(YAW)) = param_yaw_rate - drag_yaw;
}

void BicycleSlipEngine::updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                                    Eigen::Ref<state_array> state_der, const float dt)
{
  next_state = state + state_der * dt;
  next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
  next_state(S_INDEX(STEER_ANGLE)) =
      max(min(next_state(S_INDEX(STEER_ANGLE)), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  next_state(S_INDEX(STEER_ANGLE_RATE)) = state_der(S_INDEX(STEER_ANGLE));
  next_state(S_INDEX(OMEGA_Z)) = state_der(S_INDEX(YAW));
  next_state(S_INDEX(BRAKE_STATE)) =
      min(max(next_state(S_INDEX(BRAKE_STATE)), 0.0f), -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x);
}

void BicycleSlipEngine::step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
                             Eigen::Ref<state_array> state_der, const Eigen::Ref<const control_array>& control,
                             Eigen::Ref<output_array> output, const float t, const float dt)
{
  computeDynamics(state, control, state_der);
  updateState(state, next_state, state_der, dt);

  float roll = state(S_INDEX(ROLL));
  float pitch = state(S_INDEX(PITCH));
  RACER::computeStaticSettling<DYN_PARAMS_T::OutputIndex, TwoDTextureHelper<float>>(
      this->tex_helper_, next_state(S_INDEX(YAW)), next_state(S_INDEX(POS_X)), next_state(S_INDEX(POS_Y)), roll, pitch,
      output.data());
  next_state[S_INDEX(PITCH)] = pitch;
  next_state[S_INDEX(ROLL)] = roll;

  output = output_array::Zero();

  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = next_state[S_INDEX(VEL_Y)];
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(YAW)] = next_state[S_INDEX(YAW)];
  output[O_INDEX(PITCH)] = next_state[S_INDEX(PITCH)];
  output[O_INDEX(ROLL)] = next_state[S_INDEX(ROLL)];
  output[O_INDEX(STEER_ANGLE)] = next_state[S_INDEX(STEER_ANGLE)];
  output[O_INDEX(STEER_ANGLE_RATE)] = next_state[S_INDEX(STEER_ANGLE_RATE)];
  output[O_INDEX(WHEEL_FORCE_B_FL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_FR)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RR)] = 10000;
  output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
  output[O_INDEX(ACCEL_Y)] = state_der[S_INDEX(VEL_Y)];
  output[O_INDEX(OMEGA_Z)] = state_der[S_INDEX(YAW)];
}

__device__ void BicycleSlipEngine::initializeDynamics(float* state, float* control, float* output, float* theta_s,
                                                      float t_0, float dt)
{
  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* dyn_params = (DYN_PARAMS_T*)theta_s;
    *dyn_params = this->params_;
  }
  SHARED_MEM_GRD_PARAMS* shared_params = (SHARED_MEM_GRD_PARAMS*)(theta_s + shift);

  // setup memory for hidden/cell state memory
  //
  // if we are using shared memory load in the parameters
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)(shared_params + 1);
    blk_params += blockDim.x * threadIdx.z + threadIdx.x;
    steer_network_d_->initialize(&shared_params->steer_lstm_params, &shared_params->steer_output_params,
                                 &blk_params->steer_hidden_cell[0]);
    delay_network_d_->initialize(&shared_params->delay_lstm_params, &shared_params->delay_output_params,
                                 &blk_params->delay_hidden_cell[0]);
    terra_network_d_->initialize(&shared_params->terra_lstm_params, &shared_params->terra_output_params,
                                 &blk_params->terra_hidden_cell[0]);
    engine_network_d_->initialize(&shared_params->engine_lstm_params, &shared_params->engine_output_params,
                                  &blk_params->engine_hidden_cell[0]);
  }
  else
  {
    SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)(shared_params);
    blk_params += blockDim.x * threadIdx.z + threadIdx.x;
    // only setup the hidden/cell states
    steer_network_d_->initialize(nullptr, nullptr, &blk_params->steer_hidden_cell[0]);
    delay_network_d_->initialize(nullptr, nullptr, &blk_params->delay_hidden_cell[0]);
    terra_network_d_->initialize(nullptr, nullptr, &blk_params->terra_hidden_cell[0]);
    engine_network_d_->initialize(nullptr, nullptr, &blk_params->engine_hidden_cell[0]);
    __syncthreads();
  }
  for (int i = 0; i < OUTPUT_DIM && i < STATE_DIM; i++)
  {
    output[i] = state[i];
  }
}

__device__ void BicycleSlipEngine::updateState(float* state, float* next_state, float* state_der, const float dt,
                                               DYN_PARAMS_T* params_p)
{
  for (int i = threadIdx.y; i < 7; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    switch (i)
    {
      case S_INDEX(YAW):
        next_state[i] = angle_utils::normalizeAngle(next_state[i]);
        break;
      case S_INDEX(OMEGA_Z):
        next_state[i] = state_der[S_INDEX(YAW)];
      case S_INDEX(STEER_ANGLE):
        next_state[S_INDEX(STEER_ANGLE)] =
            max(min(next_state[S_INDEX(STEER_ANGLE)], params_p->max_steer_angle), -params_p->max_steer_angle);
        next_state[S_INDEX(STEER_ANGLE_RATE)] = state_der[S_INDEX(STEER_ANGLE)];
        break;
      case S_INDEX(BRAKE_STATE):
        next_state[S_INDEX(BRAKE_STATE)] =
            min(max(next_state[S_INDEX(BRAKE_STATE)], 0.0f), -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x);
    }
  }

  __syncthreads();
}

__device__ void BicycleSlipEngine::computeDynamics(float* state, float* control, float* state_der, float* theta)
{
  DYN_PARAMS_T* params_p = nullptr;

  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta;
  }
  else
  {
    params_p = &(this->params_);
  }

  // nullptr if not shared memory
  SHARED_MEM_GRD_PARAMS* params = (SHARED_MEM_GRD_PARAMS*)(theta + shift);
  SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)params;
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    // if GRD in shared them
    blk_params = (SHARED_MEM_BLK_PARAMS*)(params + 1);
  }
  blk_params = blk_params + blockDim.x * threadIdx.z + threadIdx.x;
  float* theta_s_shifted = &blk_params->theta_s[0];

  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  const float brake_cmd = -enable_brake * control[C_INDEX(THROTTLE_BRAKE)];
  const float throttle_cmd = !enable_brake * control[C_INDEX(THROTTLE_BRAKE)];
  const float delta = tanf(state[S_INDEX(STEER_ANGLE)] / params_p->wheel_angle_scale);
  const float param_yaw_rate =
      (state[S_INDEX(VEL_X)] / params_p->wheel_base) * tan(state[S_INDEX(STEER_ANGLE)] / params_p->steer_angle_scale);

  // parametric part of the brake
  state_der[S_INDEX(BRAKE_STATE)] = min(
      max((brake_cmd - state[S_INDEX(BRAKE_STATE)]) * params_p->brake_delay_constant, -params_p->max_brake_rate_neg),
      params_p->max_brake_rate_pos);

  // kinematics component
  state_der[S_INDEX(POS_X)] =
      state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]) - state[S_INDEX(VEL_Y)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] =
      state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]) + state[S_INDEX(VEL_Y)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(OMEGA_Z)] = 0.0f;

  // runs the parametric part of the steering model
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min((control[C_INDEX(STEER_CMD)] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
                  params_p->steering_constant,
              params_p->max_steer_rate),
          -params_p->max_steer_rate);

  // runs the brake model
  float* input_loc = &theta_s_shifted[DELAY_LSTM::HIDDEN_DIM];
  float* output = nullptr;
  input_loc[0] = state[S_INDEX(BRAKE_STATE)];
  input_loc[1] = brake_cmd;
  input_loc[2] = state_der[S_INDEX(BRAKE_STATE)];  // stand in for y velocity

  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    output = delay_network_d_->forward(nullptr, theta_s_shifted, &blk_params->delay_hidden_cell[0],
                                       &params->delay_lstm_params, &params->delay_output_params, 0);
  }
  else
  {
    output =
        delay_network_d_->forward(nullptr, theta_s_shifted, &blk_params->delay_hidden_cell[0],
                                  &delay_network_d_->params_, delay_network_d_->getOutputModel()->getParamsPtr(), 0);
  }
  if (threadIdx.y == 0)
  {
    state_der[S_INDEX(BRAKE_STATE)] += output[0];
  }

  // runs the engine model
  input_loc = &theta_s_shifted[ENGINE_LSTM::HIDDEN_DIM];
  input_loc[0] = throttle_cmd;
  input_loc[1] = state[S_INDEX(VEL_X)];
  input_loc[2] = state[S_INDEX(BRAKE_STATE)];
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    output = engine_network_d_->forward(nullptr, theta_s_shifted, &blk_params->engine_hidden_cell[0],
                                        &params->engine_lstm_params, &params->engine_output_params, 0);
  }
  else
  {
    output =
        engine_network_d_->forward(nullptr, theta_s_shifted, &blk_params->engine_hidden_cell[0],
                                   &engine_network_d_->params_, engine_network_d_->getOutputModel()->getParamsPtr(), 0);
  }
  const float engine_output = output[0] * 10.0f;

  // runs the steering model
  __syncthreads();  // required since we can overwrite the output before grabbing it
  input_loc = &theta_s_shifted[STEER_LSTM::HIDDEN_DIM];
  input_loc[0] = state[S_INDEX(VEL_X)];
  input_loc[1] = state[S_INDEX(STEER_ANGLE)];
  input_loc[2] = state[S_INDEX(STEER_ANGLE_RATE)];
  input_loc[3] = control[C_INDEX(STEER_CMD)];
  input_loc[4] = state_der[S_INDEX(STEER_ANGLE)];  // this is the parametric part as input
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    output = steer_network_d_->forward(nullptr, theta_s_shifted, &blk_params->steer_hidden_cell[0],
                                       &params->steer_lstm_params, &params->steer_output_params, 0);
  }
  else
  {
    output =
        steer_network_d_->forward(nullptr, theta_s_shifted, &blk_params->steer_hidden_cell[0],
                                  &steer_network_d_->params_, steer_network_d_->getOutputModel()->getParamsPtr(), 0);
  }
  if (threadIdx.y == 0)
  {
    state_der[S_INDEX(STEER_ANGLE)] += output[0] * 10.0f;
  }
  __syncthreads();  // required since we can overwrite the output before grabbing it

  // runs the terra dynamics model
  input_loc = &theta_s_shifted[TERRA_LSTM::HIDDEN_DIM];
  input_loc[0] = state[S_INDEX(VEL_X)];
  input_loc[1] = state[S_INDEX(VEL_Y)];
  input_loc[2] = state[S_INDEX(OMEGA_Z)];
  input_loc[3] = state[S_INDEX(STEER_ANGLE)];
  input_loc[4] = state[S_INDEX(STEER_ANGLE_RATE)];
  input_loc[5] = sinf(state[S_INDEX(PITCH)]) * params_p->gravity;
  input_loc[6] = sinf(state[S_INDEX(ROLL)]) * params_p->gravity;
  input_loc[7] = engine_output;
  input_loc[8] = delta;
  input_loc[9] = param_yaw_rate;
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    output = terra_network_d_->forward(nullptr, theta_s_shifted, &blk_params->terra_hidden_cell[0],
                                       &params->terra_lstm_params, &params->terra_output_params, 0);
  }
  else
  {
    output =
        terra_network_d_->forward(nullptr, theta_s_shifted, &blk_params->terra_hidden_cell[0],
                                  &terra_network_d_->params_, terra_network_d_->getOutputModel()->getParamsPtr(), 0);
  }

  const float c_delta = cosf(delta);
  const float s_delta = sinf(delta);
  const float drag_x = output[0] * 10.0f;
  const float drag_y = output[1] * 10.0f;
  const float drag_yaw = output[2] * 10.0f;

  // combine to compute state derivative
  state_der[S_INDEX(VEL_X)] = c_delta * engine_output + engine_output - drag_x * c_delta + drag_y * s_delta - drag_x;
  state_der[S_INDEX(VEL_Y)] = s_delta * engine_output - drag_x * s_delta - drag_y * c_delta - drag_y;
  state_der[S_INDEX(YAW)] = param_yaw_rate - drag_yaw;
}

__device__ void BicycleSlipEngine::step(float* state, float* next_state, float* state_der, float* control,
                                        float* output, float* theta_s, const float t, const float dt)
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
  const uint tdy = threadIdx.y;

  computeDynamics(state, control, state_der, theta_s);
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

    output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
    output[O_INDEX(BASELINK_VEL_B_Y)] = next_state[S_INDEX(VEL_Y)];
    output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
    output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
    output[O_INDEX(YAW)] = next_state[S_INDEX(YAW)];
    output[O_INDEX(PITCH)] = next_state[S_INDEX(PITCH)];
    output[O_INDEX(ROLL)] = next_state[S_INDEX(ROLL)];
    output[O_INDEX(STEER_ANGLE)] = next_state[S_INDEX(STEER_ANGLE)];
    output[O_INDEX(STEER_ANGLE_RATE)] = next_state[S_INDEX(STEER_ANGLE_RATE)];
    output[O_INDEX(WHEEL_FORCE_B_FL)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_FR)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_RL)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_RR)] = 10000;
    output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
    output[O_INDEX(ACCEL_Y)] = state_der[S_INDEX(VEL_Y)];
    output[O_INDEX(OMEGA_Z)] = state_der[S_INDEX(YAW)];
    next_state[S_INDEX(OMEGA_Z)] = state_der[S_INDEX(YAW)];
  }
}

void BicycleSlipEngine::getStoppingControl(const Eigen::Ref<const state_array>& state, Eigen::Ref<control_array> u)
{
  u[0] = -1.0;  // full brake
  u[1] = 0.0;   // no steering
}

Eigen::Quaternionf BicycleSlipEngine::attitudeFromState(const Eigen::Ref<const state_array>& state)
{
  Eigen::Quaternionf q;
  mppi::math::Euler2QuatNWU(state(S_INDEX(ROLL)), state(S_INDEX(PITCH)), state(S_INDEX(YAW)), q);
  return q;
}

Eigen::Vector3f BicycleSlipEngine::positionFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(POS_X)], state[S_INDEX(POS_Y)], 0);
}

Eigen::Vector3f BicycleSlipEngine::velocityFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(VEL_X)], state(S_INDEX(VEL_Y)), 0);
}

Eigen::Vector3f BicycleSlipEngine::angularRateFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(0, 0, state[S_INDEX(OMEGA_Z)]);
}
