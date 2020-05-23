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
};

#endif //MPPIGENERIC_MOCK_DYNAMICS_H
