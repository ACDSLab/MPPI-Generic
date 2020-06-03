/*
 * Created on Tue Jun 02 2020 by Bogdan Vlahov
 *
 */
#ifndef QUADROTOR_DYNAMICS_CUH_
#define QUADROTOR_DYNAMICS_CUH_

#include <mppi/dynamics/dynamics.cuh>

struct QuadrotorDynamicsParams {
  // TODO Fill in with actual parameters later
  float r_1 = 1;
  float tau_roll = 0.25;
  float tau_pitch = 0.25;
  float tau_yaw = 0.25;
  float mass = 1;
};

using namespace MPPI_internal;

class QuadrotorDynamics : public Dynamics<QuadrotorDynamics,
                                          QuadrotorDynamicsParams, 13, 4> {
public:
  using state_array = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        13, 4>::state_array;

  using control_array = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        13, 4>::control_array;

  using dfdx = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        13, 4>::dfdx;

  using dfdu = typename Dynamics<QuadrotorDynamics,
                                        QuadrotorDynamicsParams,
                                        13, 4>::dfdu;
  // Constructor
  QuadrotorDynamics(cudaStream_t stream = nullptr);

  // Destructor
  ~QuadrotorDynamics();

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
