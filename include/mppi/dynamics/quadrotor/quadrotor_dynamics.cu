#include <mppi/dynamics/quadrotor/quadrotor_dynamics.cuh>
#include <mppi/utils/quaternion_math.cuh>

QuadrotorDynamics::QuadrotorDynamics(cudaStream_t stream) :
Dynamics<QuadrotorDynamics, QuadrotorDynamicsParams, 13, 4>(stream) {
  this->params_ = QuadrotorDynamicsParams();
}

QuadrotorDynamics::~QuadrotorDynamics() = default;

void QuadrotorDynamics::computeDynamics(const Eigen::Ref<const state_array> &state,
                                        const Eigen::Ref<const control_array> &control,
                                        Eigen::Ref<state_array> state_der) {
  // Fill in
  state_der.block<3, 1>(0,0);
  Eigen::Matrix<float, 3, 1> x_d, v_d, angular_speed_d, u_pqr;
  Eigen::Matrix<float, 3, 1> angular_speed, v;
  Eigen::Quaternionf q_d;
  Eigen::Matrix<float, 3, 1> tau_inv;
  float u_thrust = control[4];

  // Assume quaterion is w, x, , z in state array
  Eigen::Quaternionf q(state[6], state[7], state[8], state[9]);
  u_pqr << control[0], control[1], control[2];
  v << state(3), state(4), state(5);
  angular_speed << state(10), state(11), state(12);
  tau_inv << 1 / this->params_.tau_roll,
             1 / this->params_.tau_pitch,
             1 / this->params_.tau_yaw;


  Eigen::Matrix3f dcm_lb = Eigen::Matrix3f::Identity();

  // x_d = v
  x_d = v;

  // v_d = Lvb * [0 0 T]' + g
  // TODO create DCM from quaternion
  dcm_lb = q.toRotationMatrix();
  v_d = u_thrust / this->params_.mass * dcm_lb.col(2);
  v_d(2) -= 9.81;

  // q_d = H(q) w
  mppi_math::omega2edot(u_pqr(0), u_pqr(1), u_pqr(2), q, q_d);

  // w_d = (u_pqr - w)/ tau
  // Note we assume that a low level controller makes angular velocity tracking
  // a first order system

  angular_speed_d = tau_inv.cwiseProduct(u_pqr - angular_speed);

  // Copy into state_deriv
  state_der.block<3,1>(0, 0) = x_d;
  state_der.block<3,1>(3,0) = v_d;
  state_der(6) = q_d.w();
  state_der(7) = q_d.x();
  state_der(8) = q_d.y();
  state_der(9) = q_d.z();
  state_der.block<3,1>(10, 0) = angular_speed_d;
}


void QuadrotorDynamics::updateState(Eigen::Ref<state_array> state,
                                    Eigen::Ref<state_array> state_der,
                                    float dt) {
  state += state_der * dt;

  // Renormalize quaternion
  Eigen::Quaternionf q(state[6], state[7], state[8], state[9]);
  state.block<4, 1>(6, 0) /= q.norm();
  state_der.setZero();
}

bool QuadrotorDynamics::computeGrad(const Eigen::Ref<const state_array> & state,
                                    const Eigen::Ref<const control_array>& control,
                                    Eigen::Ref<dfdx> A,
                                    Eigen::Ref<dfdu> B) {
  Eigen::Quaternionf q(state[6], state[7], state[8], state[9]);
  Eigen::Matrix3f dcm_lb;
  dcm_lb = q.toRotationMatrix();

  // I can do the math for everything except quaternions
  A.setZero();
  // x_d
  A.block<3, 3>(0, 3).setIdentity();
  // v_d I have no clue how to do this math in relation to quat

  // q_d I have no clue how this works

  // w_d
  Eigen::Vector3f tau_inv;
  tau_inv << 1 / this->params_.tau_roll,
             1 / this->params_.tau_pitch,
             1 / this->params_.tau_yaw;
  A.block<3, 3>(10, 10) = -1 * tau_inv.asDiagonal();

  B.setZero();
  // x_d is empty
  // v_d
  B.block<3, 1>(3, 3) = dcm_lb.col(2) / this->params_.mass;
  // q_d using omega2edot as reference
  B.block<4,3>(6,0) << -0.5 * q.x(), -0.5 * q.y(), -0.5 * q.z(),
                        0.5 * q.w(), -0.5 * q.z(),  0.5 * q.y(),
                        0.5 * q.z(),  0.5 * q.w(), -0.5 * q.x(),
                       -0.5 * q.y(),  0.5 * q.x(),  0.5 * q.w();
  // B(6, 0) = -0.5 * q.x();
  // B(7, 0) =  0.5 * q.w();
  // B(8, 0) =  0.5 * q.z();
  // B(9, 0) =  0.5 * q.y();
  // B(6, 1) = -0.5 * q.y();
  // B(7, 1) = -0.5 * q.z();
  // B(8, 1) =  0.5 * q.w();
  // B(9, 1) =  0.5 * q.x();
  // B(6, 2) = -0.5 * q.z();
  // B(7, 2) =  0.5 * q.y();
  // B(8, 2) = -0.5 * q.x();
  // B(9, 2) =  0.5 * q.w();
  // w_d
  B.block<3, 3>(10, 0) = tau_inv.asDiagonal();
  // Because I don't know how parts of the Jacobian dfdx are calculated, returning false
  return false;
}

__device__ void QuadrotorDynamics::computeDynamics(float* state,
                                                   float* control,
                                                   float* state_der,
                                                   float* theta) {
  //  Fill in
  float* v = &state[3];
  float* q = &state[6];
  float* w = &state[10];

  // Derivatives
  float* x_d = &state_der[0];
  float* v_d = &state_der[3];
  float* q_d = &state_der[6];
  float* w_d = &state_der[10];

  float* u_pqr = &control[0];
  float u_thrust = control[4];

  float dcm_lb[3][3];

  int i;

  // x_d = v
  for (i = threadIdx.y; i < 3; i += blockDim.y) {
    x_d[i] = v[i];
  }

  // v_d = Lvb * [0 0 T]' + g
  mppi_math::Quat2DCM(q, dcm_lb);

  for (i = threadIdx.y; i < 3; i += blockDim.y) {
    v_d[i] = u_thrust / this->params_.mass * dcm_lb[i][2];
  }
  v_d[2] -= 9.81;

  // q_d = H(q) w
  mppi_math::omega2edot(u_pqr[0], u_pqr[1], u_pqr[2], q, q_d);

  // w_d = (u - w) / tau
  w_d[0] = (u_pqr[0] - w[0]) / this->params_.tau_roll;
  w_d[1] = (u_pqr[1] - w[1]) / this->params_.tau_pitch;
  w_d[2] = (u_pqr[2] - w[2]) / this->params_.tau_yaw;
}

__device__ void QuadrotorDynamics::updateState(float* state,
                                               float* state_der,
                                               float dt) {
  int i = 0;
  for (i = threadIdx.y; i < STATE_DIM; i += blockDim.y) {
    state[i] += state_der[i] * dt;
  }

  // renormalze quaternion
  float* q = &state[6];
  float q_norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (i = threadIdx.y; i < 4; i+= blockDim.y) {
    q[i] /= q_norm;
  }
}