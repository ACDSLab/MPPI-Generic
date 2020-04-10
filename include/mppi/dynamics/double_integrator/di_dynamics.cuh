#pragma once

#ifndef DOUBLE_INTEGRATOR_CUH_
#define DOUBLE_INTEGRATOR_CUH_

#include <mppi/dynamics/dynamics.cuh>
#include <random>

struct DoubleIntegratorParams{
  float system_noise = 1;
  DoubleIntegratorParams(float noise) : system_noise(noise) {};
  DoubleIntegratorParams() = default;
  ~DoubleIntegratorParams() = default;
};

using namespace MPPI_internal;

class DoubleIntegratorDynamics : public Dynamics<DoubleIntegratorDynamics, DoubleIntegratorParams, 4, 2>
{
public:
//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DoubleIntegratorDynamics(float system_noise = 1, cudaStream_t stream = nullptr);
  ~DoubleIntegratorDynamics();

  void computeDynamics(const Eigen::Ref<const state_array> &state,
                       const Eigen::Ref<const control_array> &control,
                       Eigen::Ref<state_array> state_der);

  void computeGrad(const Eigen::Ref<const state_array> & state,
                   const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A,
                   Eigen::Ref<dfdu> B);

  void computeStateDisturbance(float dt, Eigen::Ref<state_array> state);

  void printState(float* state);
  void printState(const float* state);

  __device__ void computeDynamics(float* state,
                                  float* control,
                                  float* state_der,
                                  float* theta = nullptr);
private:
  // Random number generator for system noise
  std::mt19937 gen;  // Standard mersenne_twister_engine which will be seeded
  std::normal_distribution<float> normal_distribution;

};

#if __CUDACC__
#include "di_dynamics.cu"
#endif

#endif //!DOUBLE_INTEGRATOR_CUH_