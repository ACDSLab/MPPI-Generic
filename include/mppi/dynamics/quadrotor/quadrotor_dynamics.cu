#include <mppi/dynamics/quadrotor/quadrotor_dynamics.cuh>

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
}

bool QuadrotorDynamics::computeGrad(const Eigen::Ref<const state_array> & state,
                                    const Eigen::Ref<const control_array>& control,
                                    Eigen::Ref<dfdx> A,
                                    Eigen::Ref<dfdu> B) {
  //  Fill in
  return true;
}

__device__ void QuadrotorDynamics::computeDynamics(float* state,
                                                   float* control,
                                                   float* state_der,
                                                   float* theta) {
  //  Fill in
}