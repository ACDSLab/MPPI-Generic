/*
 * Created on Tue Jun 02 2020 by Bogdan Vlahov
 *
 */
#ifndef QUADROTOR_DYNAMICS_CUH_
#define QUADROTOR_DYNAMICS_CUH_

#include <mppi/dynamics/dynamics.cuh>

struct QuadrotorDynamicsParams {
  float tau_roll = 0.25;
  float tau_pitch = 0.25;
  float tau_yaw = 0.25;
  float mass = 1; // kg
  QuadrotorDynamicsParams(float mass_in) : mass(mass_in) {};
  QuadrotorDynamicsParams() = default;
  ~QuadrotorDynamicsParams() = default;
};

using namespace MPPI_internal;

class QuadrotorDynamics : public Dynamics<QuadrotorDynamics,
                                          QuadrotorDynamicsParams, 13, 4> {

/**
 * State for this class is defined as follows:
 *    x - position in 3D space (x, y, z) - meters
 *    v - velocity in 3D space (v_x, v_y_ v_z) - meters/sec
 *    q - quaternion (q_w, q_x, q_y, q_z)
 *    w - angular velocities (roll_dot, pitch_dot, yaw_dot) - rad/sec
 *
 * Coordinate Frame is NWU
 *
 * Control:
 *    roll_rate  - rad/sec
 *    pitch_rate - rad/sec
 *    yaw_rate   - rad/sec
 *    thrust     - Newtons
 */
public:
  using state_array = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        STATE_DIM, CONTROL_DIM>::state_array;

  using control_array = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        STATE_DIM, CONTROL_DIM>::control_array;

  using dfdx = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        STATE_DIM, CONTROL_DIM>::dfdx;

  using dfdu = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        STATE_DIM, CONTROL_DIM>::dfdu;
  // Constructor
  QuadrotorDynamics(cudaStream_t stream = 0);
  QuadrotorDynamics(std::array<float2, CONTROL_DIM> control_rngs, cudaStream_t stream = 0);

  void computeDynamics(const Eigen::Ref<const state_array>& state,
                       const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  bool computeGrad(const Eigen::Ref<const state_array> & state,
                   const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A,
                   Eigen::Ref<dfdu> B);

  void updateState(Eigen::Ref<state_array> state,
                   Eigen::Ref<state_array> s_der, float dt);

  __device__ void computeDynamics(float* state,
                                  float* control,
                                  float* state_der,
                                  float* theta = nullptr);

  __device__ void updateState(float* state, float* state_der, float dt);
};

#if __CUDACC__
#include "quadrotor_dynamics.cu"
#endif
#endif // QUADROTOR_DYNAMICS_CUH_
