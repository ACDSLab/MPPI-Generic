//
// Created by jason on 9/7/22.
//

#include "ackerman_slip.cuh"

AckermanSlip::AckermanSlip(cudaStream_t stream) : MPPI_internal::Dynamics<AckermanSlip, AckermanSlipParams>(stream)
{
  this->requires_buffer_ = true;
  tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  steer_lstm_lstm_helper_ = std::make_shared<STEER_NN>(stream);
  delay_lstm_lstm_helper_ = std::make_shared<DELAY_NN>(stream);
  engine_lstm_lstm_helper_ = std::make_shared<ENGINE_NN>(stream);
  terra_lstm_lstm_helper_ = std::make_shared<TERRA_NN>(stream);
}

AckermanSlip::AckermanSlip(std::string steer_path, std::string ackerman_path, cudaStream_t stream)
  : MPPI_internal::Dynamics<AckermanSlip, AckermanSlipParams>(stream)
{
  this->requires_buffer_ = true;
  tex_helper_ = new TwoDTextureHelper<float>(1, stream);

  steer_lstm_lstm_helper_ = std::make_shared<STEER_NN>(steer_path, stream);
  if (!fileExists(steer_path))
  {
    std::cerr << "Could not load neural net model at steer_path: " << steer_path.c_str();
    exit(-1);
  }
  cnpy::npz_t steer_param_dict = cnpy::npz_load(steer_path);
  this->params_.max_steer_rate = steer_param_dict.at("params/max_steer_rate").data<float>()[0];
  this->params_.steering_constant = steer_param_dict.at("params/steering_constant").data<float>()[0];

  delay_lstm_lstm_helper_ = std::make_shared<DELAY_NN>(stream);
  engine_lstm_lstm_helper_ = std::make_shared<ENGINE_NN>(stream);
  terra_lstm_lstm_helper_ = std::make_shared<TERRA_NN>(stream);

  if (!fileExists(ackerman_path))
  {
    std::cerr << "Could not load neural net model at ackerman_path: " << ackerman_path.c_str();
    exit(-1);
  }
  cnpy::npz_t ackerman_param_dict = cnpy::npz_load(ackerman_path);
  this->params_.gravity = ackerman_param_dict.at("params/gravity").data<float>()[0];
  this->params_.wheel_angle_scale = ackerman_param_dict.at("params/wheel_angle_scale").data<float>()[0];

  // load the delay params
  this->params_.brake_delay_constant = ackerman_param_dict.at("delay_model/params/brake_constant").data<float>()[0];
  this->params_.max_brake_rate_neg = ackerman_param_dict.at("delay_model/params/max_brake_rate_neg").data<float>()[0];
  this->params_.max_brake_rate_pos = ackerman_param_dict.at("delay_model/params/max_brake_rate_pos").data<float>()[0];

  delay_lstm_lstm_helper_->loadParams("delay_model/model", ackerman_path);
  terra_lstm_lstm_helper_->loadParams("terra_model", ackerman_path);
  engine_lstm_lstm_helper_->loadParams("engine_model", ackerman_path);
}

void AckermanSlip::initializeDynamics(const Eigen::Ref<const state_array>& state,
                                      const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output,
                                      float t_0, float dt)
{
  this->steer_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->delay_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->engine_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->terra_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  PARENT_CLASS::initializeDynamics(state, control, output, t_0, dt);
}

MPPI_internal::Dynamics<AckermanSlip, AckermanSlipParams>::state_array
AckermanSlip::stateFromMap(const std::map<std::string, float>& map)
{
  state_array s = state_array::Zero();
  // TODO
  return s;
}

void AckermanSlip::updateFromBuffer(const buffer_trajectory& buffer)
{
}

void AckermanSlip::GPUSetup()
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

void AckermanSlip::freeCudaMem()
{
  steer_lstm_lstm_helper_->freeCudaMem();
  delay_lstm_lstm_helper_->freeCudaMem();
  engine_lstm_lstm_helper_->freeCudaMem();
  terra_lstm_lstm_helper_->freeCudaMem();
  tex_helper_->freeCudaMem();
  Dynamics::freeCudaMem();
}

void AckermanSlip::paramsToDevice()
{
  // does all the internal texture updates
  tex_helper_->copyToDevice();
  PARENT_CLASS::paramsToDevice();
}

void AckermanSlip::computeDynamics(const Eigen::Ref<const state_array>& state,
                                   const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der)
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

  // kinematics component
  state_der(S_INDEX(POS_X)) =
      state(S_INDEX(VEL_X)) * cosf(state(S_INDEX(YAW))) - state(S_INDEX(VEL_Y)) * sinf(state(S_INDEX(YAW)));
  state_der(S_INDEX(POS_Y)) =
      state(S_INDEX(VEL_X)) * sinf(state(S_INDEX(YAW))) + state(S_INDEX(VEL_Y)) * cosf(state(S_INDEX(YAW)));
  state_der(S_INDEX(YAW)) = state(S_INDEX(OMEGA_Z));

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
  engine_input(2) = brake_cmd;
  ENGINE_LSTM::output_array engine_output_arr = ENGINE_LSTM::output_array::Zero();
  engine_lstm_lstm_helper_->forward(engine_input, engine_output_arr);
  float engine_output = engine_output_arr(0) * 10;

  // runs the parametric part of the steering model
  state_der(S_INDEX(STEER_ANGLE)) =
      (control(C_INDEX(STEER_CMD)) * this->params_.steer_command_angle_scale - state(S_INDEX(STEER_ANGLE))) *
      this->params_.steering_constant;
  state_der(S_INDEX(STEER_ANGLE)) =
      max(min(state_der(S_INDEX(STEER_ANGLE)), this->params_.max_steer_rate), -this->params_.max_steer_rate);

  // runs the steering model
  STEER_LSTM::input_array steer_input;
  steer_input(0) = state(S_INDEX(VEL_X));
  steer_input(1) = state(S_INDEX(VEL_Y));  // stand in for y velocity
  steer_input(2) = state(S_INDEX(STEER_ANGLE));
  steer_input(3) = state(S_INDEX(STEER_ANGLE_RATE));
  steer_input(4) = control(C_INDEX(STEER_CMD));
  steer_input(5) = state_der(S_INDEX(STEER_ANGLE));  // this is the parametric part as input
  STEER_LSTM::output_array steer_output = STEER_LSTM::output_array::Zero();
  steer_lstm_lstm_helper_->forward(steer_input, steer_output);
  state_der(S_INDEX(STEER_ANGLE)) += steer_output(0) * 10;

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
  // std::cout << "CPU input: " << terra_input.transpose() << std::endl;
  TERRA_LSTM::output_array terra_output = TERRA_LSTM::output_array::Zero();
  terra_lstm_lstm_helper_->forward(terra_input, terra_output);
  // std::cout << "CPU output: " << terra_output.transpose() << std::endl;

  float delta = tanf(state(S_INDEX(STEER_ANGLE)) / this->params_.wheel_angle_scale);

  // combine to compute state derivative
  state_der(S_INDEX(VEL_X)) = cosf(delta) * engine_output + engine_output - terra_output[0] * 10;
  state_der(S_INDEX(VEL_Y)) = sinf(delta) * engine_output - terra_output[1] * 10;
  state_der(S_INDEX(OMEGA_Z)) = terra_output[2] * 10;

  // printf("CPU delta %f, engine %f, terra %f, vel x %f\n", delta, engine_output, terra_output[0],
  // state_der(S_INDEX(VEL_X)));
}

void AckermanSlip::updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                               Eigen::Ref<state_array> state_der, const float dt)
{
  next_state = state + state_der * dt;
  next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
  next_state(S_INDEX(STEER_ANGLE)) =
      max(min(next_state(S_INDEX(STEER_ANGLE)), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  next_state(S_INDEX(STEER_ANGLE_RATE)) = state_der(S_INDEX(STEER_ANGLE));
  next_state(S_INDEX(BRAKE_STATE)) = min(max(next_state(S_INDEX(BRAKE_STATE)), 0.0f), 1.0f);
}

void AckermanSlip::step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
                        Eigen::Ref<state_array> state_der, const Eigen::Ref<const control_array>& control,
                        Eigen::Ref<output_array> output, const float t, const float dt)
{
  computeDynamics(state, control, state_der);
  updateState(state, next_state, state_der, dt);

  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);
  front_left = make_float3(front_left.x * cosf(next_state(S_INDEX(YAW))) -
                               front_left.y * sinf(next_state(S_INDEX(YAW))) + next_state(S_INDEX(POS_X)),
                           front_left.x * sinf(next_state(S_INDEX(YAW))) +
                               front_left.y * cosf(next_state(S_INDEX(YAW))) + next_state(S_INDEX(POS_Y)),
                           0);
  front_right = make_float3(front_right.x * cosf(next_state(S_INDEX(YAW))) -
                                front_right.y * sinf(next_state(S_INDEX(YAW))) + next_state(S_INDEX(POS_X)),
                            front_right.x * sinf(next_state(S_INDEX(YAW))) +
                                front_right.y * cosf(next_state(S_INDEX(YAW))) + next_state(S_INDEX(POS_Y)),
                            0);
  rear_left = make_float3(rear_left.x * cosf(next_state(S_INDEX(YAW))) - rear_left.y * sinf(next_state(S_INDEX(YAW))) +
                              next_state(S_INDEX(POS_X)),
                          rear_left.x * sinf(next_state(S_INDEX(YAW))) + rear_left.y * cosf(next_state(S_INDEX(YAW))) +
                              next_state(S_INDEX(POS_Y)),
                          0);
  rear_right = make_float3(rear_right.x * cosf(next_state(S_INDEX(YAW))) -
                               rear_right.y * sinf(next_state(S_INDEX(YAW))) + next_state(S_INDEX(POS_X)),
                           rear_right.x * sinf(next_state(S_INDEX(YAW))) +
                               rear_right.y * cosf(next_state(S_INDEX(YAW))) + next_state(S_INDEX(POS_Y)),
                           0);
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
    next_state(S_INDEX(ROLL)) = (front_roll + rear_roll) / 2;

    float left_diff = rear_left_height - front_left_height;
    left_diff = max(min(left_diff, 2.98), -2.98);
    float right_diff = rear_right_height - front_right_height;
    right_diff = max(min(right_diff, 2.98), -2.98);
    float left_pitch = asinf((left_diff) / 2.981);
    float right_pitch = asinf((right_diff) / 2.981);
    next_state(S_INDEX(PITCH)) = (left_pitch + right_pitch) / 2;
  }
  else
  {
    next_state(S_INDEX(ROLL)) = 0;
    next_state(S_INDEX(PITCH)) = 0;
  }

  if (isnan(next_state(S_INDEX(ROLL))) || isinf(next_state(S_INDEX(ROLL))) || abs(next_state(S_INDEX(ROLL))) > M_PI)
  {
    next_state(S_INDEX(ROLL)) = 4.0;
  }
  if (isnan(next_state(S_INDEX(PITCH))) || isinf(next_state(S_INDEX(PITCH))) || abs(next_state(S_INDEX(PITCH))) > M_PI)
  {
    next_state(S_INDEX(PITCH)) = 4.0;
  }

  output = output_array::Zero();

  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = next_state[S_INDEX(VEL_Y)];
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(YAW)] = next_state[S_INDEX(YAW)];
  output[O_INDEX(PITCH)] = next_state[S_INDEX(PITCH)];
  output[O_INDEX(ROLL)] = next_state[S_INDEX(ROLL)];
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
  output[O_INDEX(WHEEL_FORCE_B_FL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_FR)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RR)] = 10000;
  output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
  output[O_INDEX(ACCEL_Y)] = state_der[S_INDEX(VEL_Y)];
}

__device__ void AckermanSlip::initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0,
                                                 float dt)
{
  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD != 1)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* dyn_params = (DYN_PARAMS_T*)theta_s;
    *dyn_params = this->params_;
  }
  SHARED_MEM_GRD_PARAMS* shared_params = (SHARED_MEM_GRD_PARAMS*)(theta_s + shift);

  // setup memory for hidden/cell state memory
  //
  // if we are using shared memory load in the parameters
  if (SHARED_MEM_REQUEST_GRD != 0)
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

__device__ void AckermanSlip::updateState(float* state, float* next_state, float* state_der, const float dt)
{
}

__device__ void AckermanSlip::computeDynamics(float* state, float* control, float* state_der, float* theta)
{
  DYN_PARAMS_T* params_p = nullptr;

  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD != 1)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta;
  }
  else
  {
    params_p = &(this->params_);
  }
  const int tdy = threadIdx.y;

  // nullptr if not shared memory
  SHARED_MEM_GRD_PARAMS* params = (SHARED_MEM_GRD_PARAMS*)(theta + shift);
  SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)params;
  if (SHARED_MEM_REQUEST_GRD != 0)
  {
    // if GRD in shared them
    blk_params = (SHARED_MEM_BLK_PARAMS*)(params + 1);
  }
  // printf("shifted params %d * %d + %d =  %d\n", blockDim.x, threadIdx.z, threadIdx.x, blockDim.x * threadIdx.z +
  // threadIdx.x); printf("blk params %d\n", SHARED_MEM_REQUEST_BLK);
  blk_params = blk_params + blockDim.x * threadIdx.z + threadIdx.x;
  float* theta_s_shifted = &blk_params->theta_s[0];

  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  float brake_cmd = -enable_brake * control[C_INDEX(THROTTLE_BRAKE)];
  float throttle_cmd = !enable_brake * control[C_INDEX(THROTTLE_BRAKE)];

  state_der[S_INDEX(BRAKE_STATE)] = min(
      max((brake_cmd - state[S_INDEX(BRAKE_STATE)]) * params_p->brake_delay_constant, -params_p->max_brake_rate_neg),
      params_p->max_brake_rate_pos);

  // kinematics component
  state_der[S_INDEX(POS_X)] =
      state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]) - state[S_INDEX(VEL_Y)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] =
      state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]) + state[S_INDEX(VEL_Y)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(YAW)] = state[S_INDEX(OMEGA_Z)];

  // runs the brake model
  float* input_loc = &theta_s_shifted[DELAY_LSTM::HIDDEN_DIM * 4];
  input_loc[0] = state[S_INDEX(BRAKE_STATE)];
  input_loc[1] = brake_cmd;
  input_loc[2] = state_der[S_INDEX(BRAKE_STATE)];  // stand in for y velocity
  float* output = nullptr;

  // printf("forward call %p %p %p %p\n", theta_s_shifted, &blk_params->delay_hidden_cell[0],
  //        &params->delay_lstm_params, &params->delay_output_params);
  if (SHARED_MEM_REQUEST_GRD != 0)
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
  input_loc = &theta_s_shifted[ENGINE_LSTM::HIDDEN_DIM * 4];
  input_loc[0] = throttle_cmd;
  input_loc[1] = state[S_INDEX(VEL_X)];
  input_loc[2] = brake_cmd;
  if (SHARED_MEM_REQUEST_GRD != 0)
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
  float engine_output = output[0] * 10;

  // runs the parametric part of the steering model
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min((control[C_INDEX(STEER_CMD)] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
                  params_p->steering_constant,
              params_p->max_steer_rate),
          -params_p->max_steer_rate);

  // runs the steering model
  __syncthreads();
  input_loc = &theta_s_shifted[STEER_LSTM::HIDDEN_DIM * 4];
  input_loc[0] = state[S_INDEX(VEL_X)];
  input_loc[1] = state[S_INDEX(VEL_Y)];  // stand in for y velocity
  input_loc[2] = state[S_INDEX(STEER_ANGLE)];
  input_loc[3] = state[S_INDEX(STEER_ANGLE_RATE)];
  input_loc[4] = control[C_INDEX(STEER_CMD)];
  input_loc[5] = state_der[S_INDEX(STEER_ANGLE)];  // this is the parametric part as input
  __syncthreads();
  if (SHARED_MEM_REQUEST_GRD != 0)
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
  __syncthreads();
  if (threadIdx.y == 0)
  {
    state_der[S_INDEX(STEER_ANGLE)] += output[0] * 10;
  }
  __syncthreads();

  // runs the terra dynamics model
  input_loc = &theta_s_shifted[TERRA_LSTM::HIDDEN_DIM * 4];
  input_loc[0] = state[S_INDEX(VEL_X)];
  input_loc[1] = state[S_INDEX(VEL_Y)];
  input_loc[2] = state[S_INDEX(OMEGA_Z)];
  input_loc[3] = state[S_INDEX(STEER_ANGLE)];
  input_loc[4] = state[S_INDEX(STEER_ANGLE_RATE)];
  input_loc[5] = sinf(state[S_INDEX(PITCH)]) * this->params_.gravity;
  input_loc[6] = sinf(state[S_INDEX(ROLL)]) * this->params_.gravity;
  input_loc[7] = engine_output;
  // printf("GPU input: %f %f %f %f %f %f %f %f\n", input_loc[0], input_loc[1], input_loc[2], input_loc[3],
  //        input_loc[4], input_loc[5], input_loc[6], input_loc[7]);
  if (SHARED_MEM_REQUEST_GRD != 0)
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

  // printf("GPU terra output: %f %f %f\n", output[0], output[1], output[2]);

  float delta = tanf(state[S_INDEX(STEER_ANGLE)] / this->params_.wheel_angle_scale);

  // combine to compute state derivative
  state_der[S_INDEX(VEL_X)] = cosf(delta) * engine_output + engine_output - output[0] * 10;
  // printf("GPU delta %f, engine %f, terra %f, vel x %f\n", delta, engine_output, output[0],
  // state_der[S_INDEX(VEL_X)]);
  state_der[S_INDEX(VEL_Y)] = sinf(delta) * engine_output - output[1] * 10;
  state_der[S_INDEX(OMEGA_Z)] = output[2] * 10;
}

__device__ void AckermanSlip::step(float* state, float* next_state, float* state_der, float* control, float* output,
                                   float* theta_s, const float t, const float dt)
{
  computeDynamics(state, control, state_der, theta_s);

  DYN_PARAMS_T* params_p;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD != 1)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta_s;
  }
  else
  {
    params_p = &(this->params_);
  }
  const int tdy = threadIdx.y;

  float front_left_height = 0;
  float front_right_height = 0;
  float rear_left_height = 0;
  float rear_right_height = 0;
  // Calculate the next state
  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);

  // query the positions and set ROLL/PITCH
  front_left = make_float3(front_left.x * cosf(state[1]) - front_left.y * sinf(state[1]) + state[2],
                           front_left.x * sinf(state[1]) + front_left.y * cosf(state[1]) + state[3], 0);
  front_right = make_float3(front_right.x * cosf(state[1]) - front_right.y * sinf(state[1]) + state[2],
                            front_right.x * sinf(state[1]) + front_right.y * cosf(state[1]) + state[3], 0);
  rear_left = make_float3(rear_left.x * cosf(state[1]) - rear_left.y * sinf(state[1]) + state[2],
                          rear_left.x * sinf(state[1]) + rear_left.y * cosf(state[1]) + state[3], 0);
  rear_right = make_float3(rear_right.x * cosf(state[1]) - rear_right.y * sinf(state[1]) + state[2],
                           rear_right.x * sinf(state[1]) + rear_right.y * cosf(state[1]) + state[3], 0);

  for (int i = tdy; i < 8; i += blockDim.y)
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
        next_state[S_INDEX(BRAKE_STATE)] = min(max(next_state[S_INDEX(BRAKE_STATE)], 0.0f), 1.0f);
    }
    if (i == S_INDEX(ROLL) || i == S_INDEX(PITCH))
    {
      if (this->tex_helper_->checkTextureUse(0))
      {
        front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
        front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
        rear_left_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_left);
        rear_right_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_right);
        output[O_INDEX(BASELINK_POS_I_Z)] = (rear_right_height + rear_left_height) / 2.0f;

        // max magnitude
        if (i == S_INDEX(ROLL))
        {
          float front_diff = front_left_height - front_right_height;
          front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
          float rear_diff = rear_left_height - rear_right_height;
          rear_diff = max(min(rear_diff, 0.736 * 2), -0.736 * 2);
          float front_roll = asinf(front_diff / (0.737 * 2));
          float rear_roll = asinf(rear_diff / (0.737 * 2));
          next_state[i] = (front_roll + rear_roll) / 2;
        }
        if (i == S_INDEX(PITCH))
        {
          float left_diff = rear_left_height - front_left_height;
          left_diff = max(min(left_diff, 2.98), -2.98);
          float right_diff = rear_right_height - front_right_height;
          right_diff = max(min(right_diff, 2.98), -2.98);
          float left_pitch = asinf((left_diff) / 2.981);
          float right_pitch = asinf((right_diff) / 2.981);
          next_state[i] = (left_pitch + right_pitch) / 2;
        }
        if (isnan(next_state[i]) || isinf(next_state[i]) || fabsf(next_state[i]) > M_PIf32)
        {
          next_state[i] = 4.0;
        }
      }
      else
      {
        next_state[i] = 0;
        next_state[i] = 0;
      }
    }
  }

  __syncthreads();

  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = next_state[S_INDEX(VEL_Y)];
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(YAW)] = next_state[S_INDEX(YAW)];
  output[O_INDEX(PITCH)] = next_state[S_INDEX(PITCH)];
  output[O_INDEX(ROLL)] = next_state[S_INDEX(ROLL)];
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
  output[O_INDEX(WHEEL_FORCE_B_FL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_FR)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RR)] = 10000;
  output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
  output[O_INDEX(ACCEL_Y)] = state_der[S_INDEX(VEL_Y)];
}
