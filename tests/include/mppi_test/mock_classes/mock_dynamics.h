//
// Created by jason on 4/14/20.
//

#ifndef MPPIGENERIC_MOCK_DYNAMICS_H
#define MPPIGENERIC_MOCK_DYNAMICS_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mppi/dynamics/dynamics.cuh>

// ===== mock dynamics ====
typedef struct {
  int test = 2;
} mockDynamicsParams;

class MockDynamics : public MPPI_internal::Dynamics<MockDynamics, mockDynamicsParams, 1, 1> {
public:
  MOCK_METHOD1(bindToStream, void(cudaStream_t stream));
  MOCK_METHOD1(setParams, void(mockDynamicsParams params));
  MOCK_METHOD0(GPUSetup, void());
  MOCK_METHOD0(getControlRanges, std::array<float2, 1>());
  MOCK_METHOD0(getControlRangesRaw, void());
  MOCK_METHOD0(getParams, mockDynamicsParams());
  //MOCK_METHOD0(freeCudaMem, void());
  MOCK_METHOD0(paramsToDevice, void());
  MOCK_METHOD3(computeDynamics, void(const Eigen::Ref<const state_array>&, const Eigen::Ref<const control_array>&,
          Eigen::Ref<state_array>));
  MOCK_METHOD2(computeKinematics, void(Eigen::Ref<const state_array>&, Eigen::Ref<state_array>));
  MOCK_METHOD3(computeStateDeriv, void(const Eigen::Ref<const state_array>&, const Eigen::Ref<const control_array>&,
          Eigen::Ref<state_array>));
  MOCK_METHOD3(updateState, void(Eigen::Ref<state_array>, Eigen::Ref<state_array>, float));
  MOCK_METHOD2(enforceConstraints, void(Eigen::Ref<state_array>, Eigen::Ref<control_array>));
  MOCK_METHOD4(computeGrad, void(const Eigen::Ref<const state_array>&,
          const Eigen::Ref<const control_array>,
                  Eigen::Ref<dfdx>,
                  Eigen::Ref<dfdu>));
};

#endif //MPPIGENERIC_MOCK_DYNAMICS_H
