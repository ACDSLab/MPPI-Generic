//
// Created by jason on 12/12/22.
//

#include "bicycle_slip_hybrid.cuh"

BicycleSlipHybrid::BicycleSlipHybrid(cudaStream_t stream)
  : BicycleSlipKinematicImpl<BicycleSlipHybrid, BicycleSlipHybridParams, 12>(stream)
{
}

BicycleSlipHybrid::BicycleSlipHybrid(std::string model_path, cudaStream_t stream)
  : BicycleSlipHybrid::BicycleSlipHybrid(stream)
{
  if (!fileExists(model_path))
  {
    std::cerr << "Could not load neural net model at model_path: " << model_path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(model_path);

  // load the delay params
  this->params_.brake_delay_constant = param_dict.at("delay_model/params/constant").data<float>()[0];
  this->params_.max_brake_rate_neg = param_dict.at("delay_model/params/max_rate_neg").data<float>()[0];
  this->params_.max_brake_rate_pos = param_dict.at("delay_model/params/max_rate_pos").data<float>()[0];

  // load the steering parameters
  this->params_.max_steer_rate = param_dict.at("steer_model/params/max_rate_pos").data<float>()[0];
  this->params_.steering_constant = param_dict.at("steer_model/params/constant").data<float>()[0];

  // ax linear model components
  this->params_.c_t[0] = param_dict.at("bicycle_model/params/ax_t_low").data<float>()[0];
  this->params_.c_t[1] = param_dict.at("bicycle_model/params/ax_t").data<float>()[0];
  this->params_.c_b[0] = param_dict.at("bicycle_model/params/ax_b_low").data<float>()[0];
  this->params_.c_b[1] = param_dict.at("bicycle_model/params/ax_b").data<float>()[0];
  this->params_.c_v[0] = param_dict.at("bicycle_model/params/ax_v_low").data<float>()[0];
  this->params_.c_v[1] = param_dict.at("bicycle_model/params/ax_v").data<float>()[0];
  this->params_.c_0 = 0.0;

  // ay linear model
  this->params_.ay_angle = param_dict.at("bicycle_model/params/ay_angle").data<float>()[0];
  this->params_.ay_ax[0] = param_dict.at("bicycle_model/params/ay_ax_low").data<float>()[0];
  this->params_.ay_ax[1] = param_dict.at("bicycle_model/params/ay_ax").data<float>()[0];
  this->params_.ay_v[0] = param_dict.at("bicycle_model/params/ay_v_low").data<float>()[0];
  this->params_.ay_v[1] = param_dict.at("bicycle_model/params/ay_v").data<float>()[0];

  // omega model
  this->params_.omega_v[0] = param_dict.at("bicycle_model/params/omega_v_low").data<float>()[0];
  this->params_.omega_v[1] = param_dict.at("bicycle_model/params/omega_v").data<float>()[0];
  this->params_.steer_angle_scale = param_dict.at("bicycle_model/params/steer_angle_scale").data<float>()[0];
  this->params_.wheel_base = param_dict.at("bicycle_model/params/wheel_base").data<float>()[0];
  this->params_.gravity = param_dict.at("bicycle_model/params/gravity").data<float>()[0];

  delay_lstm_lstm_helper_->loadParams("delay_model/model", model_path);
  terra_lstm_lstm_helper_->loadParams("bicycle_model/terra_model", model_path);
  steer_lstm_lstm_helper_->loadParams("steer_model/model", model_path);
}

// TODO need carve outs for reverse gear just like in kinematic dynamics
// TODO need to inherit from elevation dynamics like kinematic since different network architectures
// TODO make a bicycle slip that can implement the generic functions without the network specializations

void BicycleSlipHybrid::computeDynamics(const Eigen::Ref<const state_array>& state,
                                        const Eigen::Ref<const control_array>& control,
                                        Eigen::Ref<state_array> state_der)
{
  state_der = state_array::Zero();
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  float throttle_cmd = !enable_brake * control(C_INDEX(THROTTLE_BRAKE));
  float brake_cmd = -enable_brake * control(C_INDEX(THROTTLE_BRAKE));
  int vx_index = abs(state(S_INDEX(VEL_X))) > 0.2;
  int vy_index = abs(state(S_INDEX(VEL_Y))) > 0.2;
  int omega_index = abs(state(S_INDEX(OMEGA_Z))) > 0.2;
  float brake_state = min(state(S_INDEX(BRAKE_STATE)), 0.3);

  // parametric brake model
  this->computeParametricDelayDeriv(state, control, state_der);

  // runs the parametric part of the steering model
  this->computeParametricSteerDeriv(state, control, state_der);

  // kinematics component
  state_der(S_INDEX(POS_X)) =
      state(S_INDEX(VEL_X)) * cosf(state(S_INDEX(YAW))) - state(S_INDEX(VEL_Y)) * sinf(state(S_INDEX(YAW)));
  state_der(S_INDEX(POS_Y)) =
      state(S_INDEX(VEL_X)) * sinf(state(S_INDEX(YAW))) + state(S_INDEX(VEL_Y)) * cosf(state(S_INDEX(YAW)));
  state_der(S_INDEX(YAW)) = state(S_INDEX(OMEGA_Z));

  // runs the parametric component of the terra dynamics model
  float throttle = this->params_.c_t[vx_index] * throttle_cmd;
  float brake = this->params_.c_b[vx_index] * brake_state * (state(S_INDEX(VEL_X)) >= 0 ? -1 : 1);
  if (vx_index == 0)
  {
    brake = this->params_.c_b[vx_index] * brake_state * -state(S_INDEX(VEL_X));
  }

  state_der(S_INDEX(VEL_X)) =
      throttle * this->params_.gear_sign + brake - this->params_.c_v[vx_index] * state(S_INDEX(VEL_X));
  if (abs(state[S_INDEX(PITCH)]) < M_PI_2f32)
  {
    state_der[S_INDEX(VEL_X)] -= this->params_.gravity * sinf(state[S_INDEX(PITCH)]);
  }

  float angle = tanf(state(S_INDEX(STEER_ANGLE)) / this->params_.ay_angle);
  state_der(S_INDEX(VEL_Y)) = angle * state_der(S_INDEX(VEL_X)) * this->params_.ay_ax[vx_index] -
                              state(S_INDEX(VEL_Y)) * this->params_.ay_v[vy_index];

  float omega_parametric = (state(S_INDEX(VEL_X)) / this->params_.wheel_base) *
                           tanf(state(S_INDEX(STEER_ANGLE)) / this->params_.steer_angle_scale);
  state_der(S_INDEX(OMEGA_Z)) = (omega_parametric - state(S_INDEX(OMEGA_Z))) / 0.02 -
                                state(S_INDEX(OMEGA_Z)) * this->params_.omega_v[omega_index];

  // runs the brake model
  if (this->params_.enable_delay_model)
  {
    DELAY_LSTM::input_array brake_input;
    brake_input(0) = state(S_INDEX(BRAKE_STATE));
    brake_input(1) = brake_cmd;
    brake_input(2) = state_der(S_INDEX(BRAKE_STATE));  // stand in for y velocity
    DELAY_LSTM::output_array brake_output = DELAY_LSTM::output_array::Zero();
    delay_lstm_lstm_helper_->forward(brake_input, brake_output);
    state_der(S_INDEX(BRAKE_STATE)) += brake_output(0);
  }

  // runs the steering model
  STEER_LSTM::input_array steer_input;
  steer_input(0) = state(S_INDEX(VEL_X)) / 20.0f;
  steer_input(1) = state(S_INDEX(STEER_ANGLE)) / 5.0f;
  steer_input(2) = state(S_INDEX(STEER_ANGLE_RATE)) / 10.0f;
  steer_input(3) = control(C_INDEX(STEER_CMD));
  steer_input(4) = state_der(S_INDEX(STEER_ANGLE)) / 10.0f;  // this is the parametric part as input
  STEER_LSTM::output_array steer_output = STEER_LSTM::output_array::Zero();
  steer_lstm_lstm_helper_->forward(steer_input, steer_output);
  state_der(S_INDEX(STEER_ANGLE)) += steer_output(0) * 10;

  // runs the terra dynamics model
  TERRA_LSTM::input_array terra_input;
  terra_input(0) = state(S_INDEX(VEL_X)) / 20.0f;
  terra_input(1) = state(S_INDEX(VEL_Y)) / 5.0f;
  terra_input(2) = state(S_INDEX(OMEGA_Z)) / 5.0f;
  terra_input(3) = throttle_cmd;
  terra_input(4) = state(S_INDEX(BRAKE_STATE));
  terra_input(5) = state(S_INDEX(STEER_ANGLE)) / 5.0f;
  terra_input(6) = state(S_INDEX(PITCH));
  terra_input(7) = state(S_INDEX(ROLL));
  terra_input(8) = this->params_.environment;
  terra_input(9) = state_der(S_INDEX(VEL_X)) / 10.0f;
  terra_input(10) = state_der(S_INDEX(VEL_Y)) / 5.0f;
  terra_input(11) = state_der(S_INDEX(OMEGA_Z)) / 10.0f;
  TERRA_LSTM::output_array terra_output = TERRA_LSTM::output_array::Zero();
  terra_lstm_lstm_helper_->forward(terra_input, terra_output);

  // combine to compute state derivative
  state_der(S_INDEX(VEL_X)) = terra_output(0) * 10.0f;
  state_der(S_INDEX(VEL_Y)) = terra_output(1) * 5.0f;
  state_der(S_INDEX(OMEGA_Z)) = terra_output(2) * 5.0f;
}

__device__ void BicycleSlipHybrid::computeDynamics(float* state, float* control, float* state_der, float* theta)
{
  // DYN_PARAMS_T* params_p = nullptr;

  // const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD / 4 + 1;
  // if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD != 1)
  // {  // Allows us to turn on or off global or shared memory version of params
  //   params_p = (DYN_PARAMS_T*)theta;
  // }
  // else
  // {
  //   params_p = &(this->params_);
  // }

  // // nullptr if not shared memory
  // SHARED_MEM_GRD_PARAMS* params = (SHARED_MEM_GRD_PARAMS*)(theta + shift);
  // SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)params;
  // if (SHARED_MEM_REQUEST_GRD != 0)
  // {
  //   // if GRD in shared them
  //   blk_params = (SHARED_MEM_BLK_PARAMS*)(params + 1);
  // }
  // blk_params = blk_params + blockDim.x * threadIdx.z + threadIdx.x;
  // float* theta_s_shifted = &blk_params->theta_s[0];

  // bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  // const float brake_cmd = -enable_brake * control[C_INDEX(THROTTLE_BRAKE)];
  // const float throttle_cmd = !enable_brake * control[C_INDEX(THROTTLE_BRAKE)];
  // const float delta = tanf(state[S_INDEX(STEER_ANGLE)] / params_p->wheel_angle_scale);

  // // parametric part of the brake
  // state_der[S_INDEX(BRAKE_STATE)] = min(
  //     max((brake_cmd - state[S_INDEX(BRAKE_STATE)]) * params_p->brake_delay_constant, -params_p->max_brake_rate_neg),
  //     params_p->max_brake_rate_pos);

  // // kinematics component
  // state_der[S_INDEX(POS_X)] =
  //     state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]) - state[S_INDEX(VEL_Y)] * sinf(state[S_INDEX(YAW)]);
  // state_der[S_INDEX(POS_Y)] =
  //     state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]) + state[S_INDEX(VEL_Y)] * cosf(state[S_INDEX(YAW)]);
  // state_der[S_INDEX(YAW)] = state[S_INDEX(OMEGA_Z)];

  // // runs the parametric part of the steering model
  // state_der[S_INDEX(STEER_ANGLE)] =
  //     max(min((control[C_INDEX(STEER_CMD)] * params_p->steer_command_angle_scale - state[S_INDEX(STEER_ANGLE)]) *
  //                 params_p->steering_constant,
  //             params_p->max_steer_rate),
  //         -params_p->max_steer_rate);

  // // runs the brake model
  // float* input_loc = &theta_s_shifted[DELAY_LSTM::HIDDEN_DIM];
  // float* output = nullptr;
  // input_loc[0] = state[S_INDEX(BRAKE_STATE)];
  // input_loc[1] = brake_cmd;
  // input_loc[2] = state_der[S_INDEX(BRAKE_STATE)];  // stand in for y velocity

  // if (SHARED_MEM_REQUEST_GRD != 0)
  // {
  //   output = delay_network_d_->forward(nullptr, theta_s_shifted, &blk_params->delay_hidden_cell[0],
  //                                      &params->delay_lstm_params, &params->delay_output_params, 0);
  // }
  // else
  // {
  //   output =
  //       delay_network_d_->forward(nullptr, theta_s_shifted, &blk_params->delay_hidden_cell[0],
  //                                 &delay_network_d_->params_, delay_network_d_->getOutputModel()->getParamsPtr(), 0);
  // }
  // if (threadIdx.y == 0)
  // {
  //   state_der[S_INDEX(BRAKE_STATE)] += output[0];
  // }

  // // runs the steering model
  // __syncthreads();  // required since we can overwrite the output before grabbing it
  // input_loc = &theta_s_shifted[STEER_LSTM::HIDDEN_DIM];
  // input_loc[0] = state[S_INDEX(VEL_X)] / 20.0f;
  // input_loc[1] = state[S_INDEX(STEER_ANGLE)] / 5.0f;
  // input_loc[2] = state[S_INDEX(STEER_ANGLE_RATE)] / 5.0f;
  // input_loc[3] = control[C_INDEX(STEER_CMD)];
  // input_loc[4] = state_der[S_INDEX(STEER_ANGLE)];  // this is the parametric part as input
  // if (SHARED_MEM_REQUEST_GRD != 0)
  // {
  //   output = steer_network_d_->forward(nullptr, theta_s_shifted, &blk_params->steer_hidden_cell[0],
  //                                      &params->steer_lstm_params, &params->steer_output_params, 0);
  // }
  // else
  // {
  //   output =
  //       steer_network_d_->forward(nullptr, theta_s_shifted, &blk_params->steer_hidden_cell[0],
  //                                 &steer_network_d_->params_, steer_network_d_->getOutputModel()->getParamsPtr(), 0);
  // }
  // if (threadIdx.y == 0)
  // {
  //   state_der[S_INDEX(STEER_ANGLE)] += output[0] * 10.0f;
  // }
  // __syncthreads();  // required since we can overwrite the output before grabbing it

  // // runs the terra dynamics model
  // input_loc = &theta_s_shifted[TERRA_LSTM::HIDDEN_DIM];
  // input_loc[0] = state[S_INDEX(VEL_X)] / 20.0f;
  // input_loc[1] = state[S_INDEX(VEL_Y)] / 5.0f;
  // input_loc[2] = state[S_INDEX(OMEGA_Z)] / 5.0f;
  // input_loc[3] = throttle_cmd;
  // input_loc[4] = state[S_INDEX(BRAKE_STATE)];
  // input_loc[5] = state[S_INDEX(STEER_ANGLE)] / 5.0f;
  // input_loc[6] = state[S_INDEX(STEER_ANGLE_RATE)] / 5.0f;
  // input_loc[7] = state[S_INDEX(PITCH)];
  // input_loc[8] = state[S_INDEX(ROLL)];
  // input_loc[9] = this->params_.environment;

  // if (SHARED_MEM_REQUEST_GRD != 0)
  // {
  //   output = terra_network_d_->forward(nullptr, theta_s_shifted, &blk_params->terra_hidden_cell[0],
  //                                      &params->terra_lstm_params, &params->terra_output_params, 0);
  // }
  // else
  // {
  //   output =
  //       terra_network_d_->forward(nullptr, theta_s_shifted, &blk_params->terra_hidden_cell[0],
  //                                 &terra_network_d_->params_, terra_network_d_->getOutputModel()->getParamsPtr(), 0);
  // }

  // const float c_delta = cosf(delta + output[3]);
  // const float s_delta = sinf(delta + output[3]);
  // const float x_accel = output[0] * 10.0f;
  // const float y_accel = output[1] * 5.0f;
  // const float yaw_accel = output[2] * 5.0f;

  // // combine to compute state derivative
  // state_der[S_INDEX(VEL_X)] = x_accel * c_delta - y_accel * s_delta + x_accel;
  // state_der[S_INDEX(VEL_Y)] = x_accel * s_delta + y_accel * c_delta + y_accel;
  // state_der[S_INDEX(OMEGA_Z)] = yaw_accel;
}
