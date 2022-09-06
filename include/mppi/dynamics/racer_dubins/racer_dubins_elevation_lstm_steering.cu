//
// Created by jason on 8/31/22.
//

#include "racer_dubins_elevation_lstm_steering.cuh"

RacerDubinsElevationLSTMSteering::RacerDubinsElevationLSTMSteering(cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering>(stream)
{
  this->requires_buffer_ = true;
  lstm_lstm_helper_ = std::make_shared<NN>(stream);
}

RacerDubinsElevationLSTMSteering::RacerDubinsElevationLSTMSteering(RacerDubinsElevationParams& params,
                                                                   cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering>(params, stream)
{
  this->requires_buffer_ = true;
  lstm_lstm_helper_ = std::make_shared<NN>(stream);
}

RacerDubinsElevationLSTMSteering::RacerDubinsElevationLSTMSteering(std::string path, cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering>(stream)
{
  this->requires_buffer_ = true;
  lstm_lstm_helper_ = std::make_shared<NN>(path, stream);

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
  bool enable_brake = control(0) < 0;
  int index = (abs(state(0)) > 0.2 && abs(state(0)) <= 3.0) + (abs(state(0)) > 3.0) * 2;

  state_der(S_INDEX(BRAKE_STATE)) =
      min(max((enable_brake * -control(C_INDEX(THROTTLE_BRAKE)) - state(S_INDEX(BRAKE_STATE))) *
                  this->params_.brake_delay_constant,
              -this->params_.max_brake_rate_neg),
          this->params_.max_brake_rate_pos);
  // applying position throttle
  float throttle = this->params_.c_t[index] * control(0);
  float brake = this->params_.c_b[index] * state(S_INDEX(BRAKE_STATE)) * (state(0) >= 0 ? -1 : 1);
  float linear_brake_slope = 0.9f * (2 / dt);
  if (abs(state(0)) <= this->params_.c_b[index] / linear_brake_slope)
  {
    throttle = this->params_.c_t[index] * max(control(0) - this->params_.low_min_throttle, 0.0f);
    brake = linear_brake_slope * state(S_INDEX(BRAKE_STATE)) * -state(0);
  }

  state_der(0) = (!enable_brake) * throttle * this->params_.gear_sign + brake - this->params_.c_v[index] * state(0) +
                 this->params_.c_0;
  if (abs(state[6]) < M_PI_2)
  {
    state_der[0] -= this->params_.gravity * sinf(state[S_INDEX(PITCH)]);
  }
  state_der(1) = (state(0) / this->params_.wheel_base) * tan(state(4) / this->params_.steer_angle_scale[index]);
  state_der(2) = state(0) * cosf(state(1));
  state_der(3) = state(0) * sinf(state(1));
  state_der(4) = (control(1) * this->params_.steer_command_angle_scale - state(4)) * this->params_.steering_constant;
  state_der(4) = max(min(state_der(4), this->params_.max_steer_rate), -this->params_.max_steer_rate);

  LSTM::input_array input;
  input(0) = state(S_INDEX(VEL_X));
  input(1) = 0.0f;  // stand in for y velocity
  input(2) = state(S_INDEX(STEER_ANGLE));
  input(3) = state(S_INDEX(STEER_ANGLE_RATE));
  input(4) = control(C_INDEX(STEER_CMD));
  input(5) = state_der(S_INDEX(STEER_ANGLE));  // this is the parametric part as input
  LSTM::output_array nn_output = LSTM::output_array::Zero();
  lstm_lstm_helper_->forward(input, nn_output);
  state_der(S_INDEX(STEER_ANGLE)) += nn_output(0) * 10;

  // Integrate using racer_dubins updateState
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

  // Setup output
  float yaw = next_state[S_INDEX(YAW)];
  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
  output[O_INDEX(BASELINK_POS_I_X)] = next_state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = next_state[S_INDEX(POS_Y)];
  output[O_INDEX(BASELINK_POS_I_Z)] = 0;
  output[O_INDEX(YAW)] = yaw;
  output[O_INDEX(PITCH)] = pitch;
  output[O_INDEX(ROLL)] = roll;
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
  output[O_INDEX(WHEEL_FORCE_B_FL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_FR)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RL)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RR)] = 10000;
  // output[O_INDEX(CENTER_POS_I_X)] = output[O_INDEX(BASELINK_POS_I_X)];  // TODO
  // output[O_INDEX(CENTER_POS_I_Y)] = output[O_INDEX(BASELINK_POS_I_Y)];
  // output[O_INDEX(CENTER_POS_I_Z)] = 0;
  output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
}

__device__ void RacerDubinsElevationLSTMSteering::initializeDynamics(float* state, float* control, float* output,
                                                                     float* theta_s, float t_0, float dt)
{
  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD != 1)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* shared_params = (DYN_PARAMS_T*)theta_s;
    *shared_params = this->params_;
  }
  network_d_->initialize(theta_s + shift);
}

__device__ inline void RacerDubinsElevationLSTMSteering::step(float* state, float* next_state, float* state_der,
                                                              float* control, float* output, float* theta_s,
                                                              const float t, const float dt)
{
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

  // Compute dynamics
  bool enable_brake = control[0] < 0;
  int index = (fabsf(state[S_INDEX(VEL_X)]) > 0.2 && fabsf(state[S_INDEX(VEL_X)]) <= 3.0) +
              (fabsf(state[S_INDEX(VEL_X)]) > 3.0) * 2;

  state_der[S_INDEX(BRAKE_STATE)] =
      min(max((enable_brake * -control[C_INDEX(THROTTLE_BRAKE)] - state[S_INDEX(BRAKE_STATE)]) *
                  this->params_.brake_delay_constant,
              -this->params_.max_brake_rate_neg),
          this->params_.max_brake_rate_pos);

  // applying position throttle
  float throttle = params_p->c_t[index] * control[0];
  float brake = params_p->c_b[index] * state[S_INDEX(BRAKE_STATE)] * (state[S_INDEX(VEL_X)] >= 0 ? -1 : 1);
  float linear_brake_slope = 0.9f * (2 / dt);
  if (abs(state[S_INDEX(VEL_X)]) <= params_p->c_b[index] / linear_brake_slope)
  {
    throttle = params_p->c_t[index] * max(control[0] - params_p->low_min_throttle, 0.0f);
    brake = linear_brake_slope * state[S_INDEX(BRAKE_STATE)] * -state[S_INDEX(VEL_X)];
  }

  if (threadIdx.y == 0)
  {
    state_der[S_INDEX(VEL_X)] = (!enable_brake) * throttle * this->params_.gear_sign + brake -
                                params_p->c_v[index] * state[S_INDEX(VEL_X)] + params_p->c_0;
    if (fabsf(state[S_INDEX(PITCH)]) < M_PI_2f32)
    {
      state_der[S_INDEX(VEL_X)] -= params_p->gravity * sinf(state[S_INDEX(PITCH)]);
    }
  }
  state_der[S_INDEX(YAW)] = (state[S_INDEX(VEL_X)] / params_p->wheel_base) *
                            tan(state[S_INDEX(STEER_ANGLE)] / params_p->steer_angle_scale[index]);
  state_der[S_INDEX(POS_X)] = state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] = state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(STEER_ANGLE)] =
      max(min((control[1] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
                  params_p->steering_constant,
              params_p->max_steer_rate),
          -params_p->max_steer_rate);

  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD / 4 + 1;
  // loads in the input to the network
  float* input_loc = network_d_->getInputLocation(theta_s + shift);
  if (threadIdx.y == 0)
  {
    input_loc[0] = state[S_INDEX(VEL_X)];
    input_loc[1] = 0.0f;  // filler for VEL_Y
    input_loc[2] = state[S_INDEX(STEER_ANGLE)];
    input_loc[3] = state[S_INDEX(STEER_ANGLE_RATE)];
    input_loc[4] = control[C_INDEX(STEER_CMD)];
    input_loc[5] = state_der[S_INDEX(STEER_ANGLE)];  // this is the parametric part as input
  }
  __syncthreads();
  // runs the network
  float* nn_output = network_d_->forward(nullptr, theta_s + shift);
  // copies the results of the network to state derivative
  if (threadIdx.y == 0)
  {
    state_der[S_INDEX(STEER_ANGLE)] += nn_output[0] * 10;
  }
  __syncthreads();

  // Calculate the next state
  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);

  float front_left_height = 0;
  float front_right_height = 0;
  float rear_left_height = 0;
  float rear_right_height = 0;
  front_left = make_float3(front_left.x * cosf(state[1]) - front_left.y * sinf(state[1]) + state[2],
                           front_left.x * sinf(state[1]) + front_left.y * cosf(state[1]) + state[3], 0);
  front_right = make_float3(front_right.x * cosf(state[1]) - front_right.y * sinf(state[1]) + state[2],
                            front_right.x * sinf(state[1]) + front_right.y * cosf(state[1]) + state[3], 0);
  rear_left = make_float3(rear_left.x * cosf(state[1]) - rear_left.y * sinf(state[1]) + state[2],
                          rear_left.x * sinf(state[1]) + rear_left.y * cosf(state[1]) + state[3], 0);
  rear_right = make_float3(rear_right.x * cosf(state[1]) - rear_right.y * sinf(state[1]) + state[2],
                           rear_right.x * sinf(state[1]) + rear_right.y * cosf(state[1]) + state[3], 0);

  // Set to 8 as the last 2 states do not do euler integration
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
            max(min(next_state[S_INDEX(STEER_ANGLE)], this->params_.max_steer_angle), -this->params_.max_steer_angle);
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
          if (isnan(next_state[i]) || isinf(next_state[i]) || fabsf(next_state[i]) > M_PIf32)
          {
            next_state[i] = 4.0;
          }
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

  // Fill in output
  output[O_INDEX(BASELINK_VEL_B_X)] = next_state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
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
  // output[O_INDEX(CENTER_POS_I_X)] = output[O_INDEX(BASELINK_POS_I_X)];  // TODO
  // output[O_INDEX(CENTER_POS_I_Y)] = output[O_INDEX(BASELINK_POS_I_Y)];
  // output[O_INDEX(CENTER_POS_I_Z)] = 0;
  output[O_INDEX(ACCEL_X)] = state_der[S_INDEX(VEL_X)];
}

void RacerDubinsElevationLSTMSteering::updateFromBuffer(const buffer_trajectory& buffer)
{
  NN::init_buffer init_buffer;
  if (buffer.find("VEL_X") == buffer.end() || buffer.find("VEL_Y") == buffer.end() ||
      buffer.find("STEER_ANGLE") == buffer.end() || buffer.find("STEER_ANGLE_RATE") == buffer.end() ||
      buffer.find("STEER_CMD") == buffer.end())
  {
    std::cout << "WARNING: not using init buffer" << std::endl;
    for (const auto& it : buffer)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return;
  }

  init_buffer.row(0) = buffer.at("VEL_X");
  init_buffer.row(1) = buffer.at("VEL_Y");
  init_buffer.row(2) = buffer.at("STEER_ANGLE");
  init_buffer.row(3) = buffer.at("STEER_ANGLE_RATE");
  init_buffer.row(4) = buffer.at("STEER_CMD");

  lstm_lstm_helper_->initializeLSTM(init_buffer);
}

void RacerDubinsElevationLSTMSteering::initializeDynamics(const Eigen::Ref<const state_array>& state,
                                                          const Eigen::Ref<const control_array>& control,
                                                          Eigen::Ref<output_array> output, float t_0, float dt)
{
  this->lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  PARENT_CLASS::initializeDynamics(state, control, output, t_0, dt);
}
