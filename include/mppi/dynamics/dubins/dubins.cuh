#pragma once

#ifndef DUBINS_CUH_
#define DUBINS_CUH_

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/angle_utils.cuh>
#include <random>

struct DubinsParams{
  DubinsParams() = default;
  ~DubinsParams() = default;
};

using namespace MPPI_internal;
/**
 * state: x, y, theta
 * control: forward velocity, angular velocity
 */
class DubinsDynamics : public Dynamics<DubinsDynamics, DubinsParams, 3, 2>
{
public:
  DubinsDynamics(cudaStream_t stream = nullptr);

  void computeDynamics(const Eigen::Ref<const state_array> &state,
                       const Eigen::Ref<const control_array> &control,
                       Eigen::Ref<state_array> state_der);

  void updateState(Eigen::Ref<state_array> state,
                   Eigen::Ref<state_array> state_der, const float dt);

  __device__ void updateState(float* state, float* state_der, const float dt);

  state_array interpolateState(const Eigen::Ref<state_array> state_1,
                               const Eigen::Ref<state_array> state_2, const double alpha);

    bool computeGrad(const Eigen::Ref<const state_array> & state,
                   const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A,
                   Eigen::Ref<dfdu> B);

  __device__ void computeDynamics(float* state,
                                  float* control,
                                  float* state_der,
                                  float* theta = nullptr);
private:

};

#if __CUDACC__
#include "dubins.cu"
#endif

#endif //!DUBINS_CUH_
