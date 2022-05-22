#include <mppi/dynamics/racer_suspension/racer_suspension.cuh>
#include <mppi/utils/eigen_type_conversions.h>
#include <mppi/utils/math_utils.h>

void RacerSuspension::GPUSetup()
{
  auto* derived = static_cast<PARENT_CLASS*>(this);
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
  PARENT_CLASS::paramsToDevice();
}

// combined computeDynamics & updateState
void RacerSuspension::step(Eigen::Ref<state_array> state, Eigen::Ref<const control_array> control, const float dt)
{
  state_array x_dot;
  Eigen::Matrix3f omegaJac;
  computeStateDeriv(state, control, x_dot, nullptr, &omegaJac);
  // approximate implicit euler for angular rate states
  Eigen::Vector3f deltaOmega =
      (Eigen::Matrix3f::Identity() - dt * omegaJac).inverse() * dt * x_dot.segment<3>(S_INDEX(OMEGA_B_X));
  state_array delta_x = x_dot * dt;
  delta_x.segment<3>(S_INDEX(OMEGA_B_X)) = deltaOmega;
  state += delta_x;
  float q_norm = state.segment<4>(S_INDEX(ATTITUDE_QW)).norm();
  state.segment<4>(S_INDEX(ATTITUDE_QW)) /= q_norm;
}

void RacerSuspension::updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt)
{
  state += state_der * dt;
  float q_norm = state.segment<4>(S_INDEX(ATTITUDE_QW)).norm();
  state.segment<4>(S_INDEX(ATTITUDE_QW)) /= q_norm;
  state_der.setZero();
}

__device__ void RacerSuspension::updateState(float* state, float* state_der, const float dt)
{
  unsigned int i;
  unsigned int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < PARENT_CLASS::STATE_DIM; i += blockDim.y)
  {
    state[i] += state_der[i] * dt;
  }
  // TODO renormalize quaternion
}

__device__ __host__ static float stribeck_friction(float v, float mu_s, float v_slip, float& partial_mu_partial_v)
{
    float mu = v / v_slip * mu_s;
    partial_mu_partial_v = 0;
    if (mu > mu_s)
    {
      return mu_s;
    }
    if (mu < -mu_s)
    {
      return -mu_s;
    }
    partial_mu_partial_v = mu_s / v_slip;
    return mu;
}

__device__ __host__ void RacerSuspension::computeStateDeriv(const Eigen::Ref<const state_array>& state,
                                                            const Eigen::Ref<const control_array>& control,
                                                            Eigen::Ref<state_array> state_der,
                                                            Eigen::Ref<output_array>* output,
                                                            Eigen::Matrix3f* omegaJacobian)
{
  Eigen::Vector3f p_I = state.segment<3>(S_INDEX(P_I_X));
  Eigen::Vector3f v_I = state.segment<3>(S_INDEX(V_I_X));
  Eigen::Vector3f omega = state.segment<3>(S_INDEX(OMEGA_B_X));
  Eigen::Quaternionf q(state[S_INDEX(ATTITUDE_QW)], state[S_INDEX(ATTITUDE_QX)], state[S_INDEX(ATTITUDE_QY)],
                       state[S_INDEX(ATTITUDE_QZ)]);
  Eigen::Matrix3f R = q.toRotationMatrix();
  float tan_delta = tan(state[S_INDEX(STEER_ANGLE)]);
  Eigen::Matrix3f Rdot = R * mppi::math::skewSymmetricMatrix(omega);

  // linear engine model
  float vel_x = (R.transpose() * v_I)[0];
  float throttle = max(0.0, control[C_INDEX(THROTTLE_BRAKE)]);
  float brake = max(0.0, -control[C_INDEX(THROTTLE_BRAKE)]);
  float acc = params_.c_t * throttle - copysign(params_.c_b * brake, vel_x) - params_.c_v * vel_x + params_.c_0;
  float propulsion_force = params_.mass * acc;

  Eigen::Vector3f f_B = Eigen::Vector3f::Zero();
  Eigen::Vector3f tau_B = Eigen::Vector3f::Zero();
  Eigen::Matrix3f tau_B_jac = Eigen::Matrix3f::Zero();
  Eigen::Vector3f f_r_B[4];
  // for each wheel
  for (int i = 0; i < 4; i++)
  {
    // compute suspension from elevation map
    Eigen::Vector3f p_w_nom_B_i =
        cudaToEigen(params_.wheel_pos_wrt_base_link[i]) - cudaToEigen(params_.cg_pos_wrt_base_link);
    Eigen::Vector3f p_w_nom_I_i = p_I + R * p_w_nom_B_i;
    float h_i = 0;
    Eigen::Vector3f n_I_i(0, 0, 1);
    // if (tex_helper_->checkTextureUse(TEXTURE_ELEVATION_MAP))
    // {
    //   float4 wheel_elev = tex_helper_->queryTextureAtWorldPose(TEXTURE_ELEVATION_MAP, EigenToCuda(p_w_nom_I_i));
    //   h_i = wheel_elev.w;
    //   // std::cout << "h_i " << h_i << std::endl;
    //   n_I_i = Eigen::Vector3f(wheel_elev.x, wheel_elev.y, wheel_elev.z);
    // }
    Eigen::Vector3f p_c_I_i(p_w_nom_I_i[0], p_w_nom_I_i[1], h_i);
    float l_i = p_w_nom_I_i[2] - p_c_I_i[2];
    Eigen::Vector3f p_dot_w_nom_I_i = v_I + Rdot * p_w_nom_B_i;
    Eigen::Matrix3f p_dot_w_nom_I_i_Jac = R * Eigen::Matrix3f::Identity().colwise().cross(p_w_nom_B_i);
    float h_dot_i = -n_I_i[0] / n_I_i[2] * p_dot_w_nom_I_i[0] - n_I_i[1] / n_I_i[2] * p_dot_w_nom_I_i[1];
    Eigen::RowVector3f h_dot_i_Jac =
        -n_I_i[0] / n_I_i[2] * p_dot_w_nom_I_i_Jac.row(0) - n_I_i[1] / n_I_i[2] * p_dot_w_nom_I_i_Jac.row(1);
    float l_dot_i = p_dot_w_nom_I_i[2] - h_dot_i;
    Eigen::RowVector3f l_dot_i_Jac = p_dot_w_nom_I_i_Jac.row(2) - h_dot_i_Jac;

    float f_k_i = -params_.k_s[i] * (l_i - params_.l_0[i]) - params_.c_s[i] * l_dot_i;
    Eigen::RowVector3f f_k_i_Jac = -params_.c_s[i] * l_dot_i_Jac;
    if (f_k_i < 0)
    {
      f_k_i = 0;
      f_k_i_Jac = Eigen::RowVector3f::Zero();
    }

    // contact frame
    Eigen::Vector3f p_dot_c_I_i(p_dot_w_nom_I_i[0], p_dot_w_nom_I_i[1], h_dot_i);
    Eigen::Matrix3f p_dot_c_I_i_Jac;
    p_dot_c_I_i_Jac.row(0) = p_dot_w_nom_I_i_Jac.row(0);
    p_dot_c_I_i_Jac.row(1) = p_dot_w_nom_I_i_Jac.row(1);
    p_dot_c_I_i_Jac.row(2) = h_dot_i_Jac;
    float delta_i;
    if (i == RacerSuspensionParams::WHEEL_FRONT_LEFT)
    {
      delta_i = atan(params_.wheel_base * tan_delta / (params_.wheel_base - tan_delta * params_.width / 2));
    }
    else if (i == RacerSuspensionParams::WHEEL_FRONT_RIGHT)
    {
      delta_i = atan(params_.wheel_base * tan_delta / (params_.wheel_base + tan_delta * params_.width / 2));
    }
    else
    {  // rear wheels
      delta_i = 0;
    }

    Eigen::Vector3f n_B_i = R.transpose() * n_I_i;
    Eigen::Vector3f wheel_dir_B_i(cos(delta_i), sin(delta_i), 0);
    Eigen::Vector3f s_B_i = n_B_i.cross(wheel_dir_B_i).normalized();
    Eigen::Vector3f t_B_i = s_B_i.cross(n_B_i);
    Eigen::Matrix3f R_C_i_to_B;
    R_C_i_to_B.col(0) = t_B_i;
    R_C_i_to_B.col(1) = s_B_i;
    R_C_i_to_B.col(2) = n_B_i;

    // contact velocity
    Eigen::Vector3f p_dot_c_B_i = R.transpose() * p_dot_c_I_i;
    Eigen::Matrix3f p_dot_c_B_i_Jac = R.transpose() * p_dot_c_I_i_Jac;
    float v_w_t_i = t_B_i.dot(p_dot_c_B_i);
    float v_w_s_i = s_B_i.dot(p_dot_c_B_i);
    Eigen::RowVector3f v_w_s_i_Jac = s_B_i.transpose() * p_dot_c_B_i_Jac;

    // compute contact forces
    float f_n_i = f_k_i;
    float partial_mu_s_partial_v;
    float mu_s = stribeck_friction(v_w_s_i, params_.mu, params_.v_slip, partial_mu_s_partial_v);
    float f_s_i = -mu_s * f_n_i;
    float f_t_i = max(-params_.mu * f_n_i, min(propulsion_force / 4, params_.mu * f_n_i));
    Eigen::RowVector3f f_n_i_Jac = f_k_i_Jac;
    Eigen::RowVector3f f_t_i_Jac = Eigen::RowVector3f::Zero();
    if (propulsion_force / 4 > params_.mu * f_n_i)
    {
      f_t_i_Jac = params_.mu * f_n_i_Jac;
    }
    else if (propulsion_force / 4 < -params_.mu * f_n_i)
    {
      f_t_i_Jac = -params_.mu * f_n_i_Jac;
    }
    Eigen::RowVector3f f_s_i_Jac = -f_n_i * partial_mu_s_partial_v * v_w_s_i_Jac - mu_s * f_n_i_Jac;

    // contact force & location
    Eigen::Vector3f f_r_C_i(f_t_i, f_s_i, f_n_i);
    Eigen::Matrix3f f_r_C_i_Jac;
    f_r_C_i_Jac.row(0) = f_t_i_Jac;
    f_r_C_i_Jac.row(1) = f_s_i_Jac;
    f_r_C_i_Jac.row(2) = f_n_i_Jac;
    f_r_B[i] = R_C_i_to_B * f_r_C_i;
    Eigen::Matrix3f f_r_B_i_Jac = R_C_i_to_B = f_r_C_i_Jac;
    Eigen::Vector3f p_c_B_i = R.transpose() * (p_c_I_i - p_I);

    // accumulate forces & moments
    f_B += f_r_B[i];
    tau_B += p_c_B_i.cross(f_r_B[i]);
    tau_B_jac += -f_r_B_i_Jac.colwise().cross(p_c_B_i);

    if (output)
    {
      (*output)[O_INDEX(WHEEL_POS_I_FL_X) + i * 2] = p_w_nom_I_i[0];
      (*output)[O_INDEX(WHEEL_POS_I_FL_Y) + i * 2] = p_w_nom_I_i[1];
      (*output)[O_INDEX(WHEEL_FORCE_B_FL_X) + i * 3] = f_r_B[i][0];
      (*output)[O_INDEX(WHEEL_FORCE_B_FL_Y) + i * 3] = f_r_B[i][1];
      (*output)[O_INDEX(WHEEL_FORCE_B_FL_Z) + i * 3] = f_r_B[i][2];
    }
  }

  Eigen::Vector3f g(0, 0, params_.gravity);  // TODO gravity is negative to match dubins model

  state_der.segment<3>(S_INDEX(P_I_X)) = v_I;
  state_der.segment<3>(S_INDEX(V_I_X)) = 1 / params_.mass * R * f_B + g;
  Eigen::Quaternionf qdot;
  qdot.coeffs() = 0.5 * (q * Eigen::Quaternionf(0, omega[0], omega[1], omega[2])).coeffs();
  state_der[S_INDEX(ATTITUDE_QW)] = qdot.w();
  state_der[S_INDEX(ATTITUDE_QX)] = qdot.x();
  state_der[S_INDEX(ATTITUDE_QY)] = qdot.y();
  state_der[S_INDEX(ATTITUDE_QZ)] = qdot.z();
  Eigen::Vector3f J_diag(params_.Jxx, params_.Jyy, params_.Jzz);
  Eigen::Vector3f J_inv_diag(1/params_.Jxx, 1/params_.Jyy, 1/params_.Jzz);
  state_der.segment<3>(S_INDEX(OMEGA_B_X)) = J_inv_diag.cwiseProduct(J_diag.cwiseProduct(omega).cross(omega) + tau_B);
  if (omegaJacobian)
  {
    Eigen::Matrix3f J = J_diag.asDiagonal();
    Eigen::Matrix3f Jwxw_jac =
        J.colwise().cross(omega) - Eigen::Matrix3f::Identity().colwise().cross(J_diag.cwiseProduct(omega));
    *omegaJacobian = J_inv_diag.asDiagonal() * (Jwxw_jac + tau_B_jac);
  }

  // Steering actuator model
  float steer = control[C_INDEX(STEER_CMD)] / params_.steer_command_angle_scale;
  state_der[S_INDEX(STEER_ANGLE)] = params_.steering_constant * (steer - state[S_INDEX(STEER_ANGLE)]);

  if (output)
  {
    Eigen::Vector3f COM_v_B = R.transpose() * v_I;
    Eigen::Vector3f p_base_link_in_B = -cudaToEigen(params_.cg_pos_wrt_base_link);
    Eigen::Vector3f base_link_v_B = COM_v_B + omega.cross(p_base_link_in_B);
    (*output)[O_INDEX(BASELINK_VEL_B_X)] = base_link_v_B[0];
    (*output)[O_INDEX(BASELINK_VEL_B_X)] = base_link_v_B[1];
    (*output)[O_INDEX(BASELINK_VEL_B_X)] = base_link_v_B[2];
    Eigen::Vector3f base_link_p_I = p_I + R * p_base_link_in_B;
    (*output)[O_INDEX(BASELINK_POS_I_X)] = base_link_p_I[0];
    (*output)[O_INDEX(BASELINK_POS_I_Y)] = base_link_p_I[1];
    (*output)[O_INDEX(BASELINK_POS_I_Z)] = base_link_p_I[2];
    (*output)[O_INDEX(OMEGA_B_X)] = omega[0];
    (*output)[O_INDEX(OMEGA_B_Y)] = omega[1];
    (*output)[O_INDEX(OMEGA_B_Z)] = omega[2];
    float roll, pitch, yaw;
    mppi::math::Quat2EulerNWU(q, roll, pitch, yaw);
    (*output)[O_INDEX(YAW)] = yaw;
    (*output)[O_INDEX(PITCH)] = pitch;
    (*output)[O_INDEX(ROLL)] = roll;
    (*output)[O_INDEX(ATTITUDE_QW)] = q.w();
    (*output)[O_INDEX(ATTITUDE_QX)] = q.x();
    (*output)[O_INDEX(ATTITUDE_QY)] = q.y();
    (*output)[O_INDEX(ATTITUDE_QZ)] = q.z();
    (*output)[O_INDEX(STEER_ANGLE)] = state[S_INDEX(STEER_ANGLE)];
    (*output)[O_INDEX(STEER_ANGLE_RATE)] = state_der[S_INDEX(STEER_ANGLE)];
    (*output)[O_INDEX(CENTER_POS_I_X)] = (*output)[O_INDEX(BASELINK_POS_I_X)];  // TODO
    (*output)[O_INDEX(CENTER_POS_I_Y)] = (*output)[O_INDEX(BASELINK_POS_I_Y)];
    (*output)[O_INDEX(CENTER_POS_I_Z)] = (*output)[O_INDEX(BASELINK_POS_I_Z)];
  }
}

__device__ void RacerSuspension::computeStateDeriv(float* state, float* control, float* state_der, float* theta_s,
                                                   float* output)
{
  (void)theta_s;
  Eigen::Map<state_array> state_v(state);
  Eigen::Map<control_array> control_v(control);
  Eigen::Map<state_array> state_der_v(state_der);
  if (output)
  {
    Eigen::Map<output_array> output_v(output);
    Eigen::Ref<output_array> output_r(output_v);
    computeStateDeriv(state_v, control_v, state_der_v, &output_r);
  }
  else
  {
    computeStateDeriv(state_v, control_v, state_der_v, nullptr);
  }
  // for (int i = 0; i < PARENT_CLASS::STATE_DIM; i++)
  // {
  //   state_der[i] = 0;
  // }
  // state_der[S_INDEX(V_I_X)] = control[1];
  // if (output)
  // {
  //   for (int i = 0; i < PARENT_CLASS::OUTPUT_DIM; i++)
  //   {
  //     output[i] = 0;
  //   }
  // }
}

Eigen::Quaternionf RacerSuspension::attitudeFromState(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Quaternionf(state[S_INDEX(ATTITUDE_QW)], state[S_INDEX(ATTITUDE_QX)], state[S_INDEX(ATTITUDE_QY)],
                            state[S_INDEX(ATTITUDE_QZ)]);
}

Eigen::Vector3f RacerSuspension::positionFromState(const Eigen::Ref<const state_array>& state)
{
  Eigen::Vector3f p_COM = state.segment<3>(S_INDEX(P_I_X));
  Eigen::Quaternionf q = attitudeFromState(state);
  return p_COM - q * cudaToEigen(params_.cg_pos_wrt_base_link);
}

Eigen::Vector3f RacerSuspension::velocityFromState(const Eigen::Ref<const state_array>& state)
{
  Eigen::Vector3f COM_v_I = state.segment<3>(S_INDEX(V_I_X));
  Eigen::Quaternionf q_B_to_I = attitudeFromState(state);
  Eigen::Vector3f COM_v_B = q_B_to_I.conjugate() * COM_v_I;
  Eigen::Vector3f omega = state.segment<3>(S_INDEX(OMEGA_B_X));
  Eigen::Vector3f p_base_link_in_B = -cudaToEigen(params_.cg_pos_wrt_base_link);
  Eigen::Vector3f base_link_v_B = COM_v_B + omega.cross(p_base_link_in_B);
  return base_link_v_B;
}

Eigen::Vector3f RacerSuspension::angularRateFromState(const Eigen::Ref<const state_array>& state)
{
  return state.segment<3>(S_INDEX(OMEGA_B_X));
}

RacerSuspension::state_array RacerSuspension::stateFromOdometry(const Eigen::Quaternionf& q_B_to_I,
                                                                const Eigen::Vector3f& pos_base_link_I,
                                                                const Eigen::Vector3f& vel_base_link_B,
                                                                const Eigen::Vector3f& omega_B)
{
  state_array s;
  s.setZero();
  s[S_INDEX(ATTITUDE_QW)] = q_B_to_I.w();
  s[S_INDEX(ATTITUDE_QX)] = q_B_to_I.x();
  s[S_INDEX(ATTITUDE_QY)] = q_B_to_I.y();
  s[S_INDEX(ATTITUDE_QZ)] = q_B_to_I.z();
  s.segment<3>(S_INDEX(OMEGA_B_X)) = omega_B;
  Eigen::Vector3f p_COM_wrt_base_link = cudaToEigen(params_.cg_pos_wrt_base_link);
  Eigen::Vector3f p_I = pos_base_link_I + q_B_to_I * p_COM_wrt_base_link;
  s.segment<3>(S_INDEX(P_I_X)) = p_I;
  Eigen::Vector3f COM_v_B = vel_base_link_B + omega_B.cross(p_COM_wrt_base_link);
  Eigen::Vector3f COM_v_I = q_B_to_I * COM_v_B;
  s.segment<3>(S_INDEX(V_I_X)) = COM_v_I;
  return s;
}
