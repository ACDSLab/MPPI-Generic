//
// Created by jason on 4/14/20.
//

#ifndef MPPIGENERIC_MOCK_CONTROLLER_H
#define MPPIGENERIC_MOCK_CONTROLLER_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mppi/controllers/controller.cuh>
#include <mppi_test/mock_classes/mock_dynamics.h>
#include <mppi_test/mock_classes/mock_costs.h>

// ===== mock controller ====
class MockController : public Controller<MockDynamics, MockCost, 100, 500, 32, 2> {
public:
  MOCK_METHOD0(resetControls, void());
  MOCK_METHOD1(computeFeedbackGains, void(const Eigen::Ref<const state_array>& state));
  MOCK_METHOD1(slideControlSequence, void(int stride));
  MOCK_METHOD1(computeControl, void(const Eigen::Ref<const state_array>& state));
  MOCK_METHOD0(getControlSeq, control_trajectory());
  MOCK_METHOD0(getStateSeq, state_trajectory());
  MOCK_METHOD0(getFeedbackGains, feedback_gain_trajectory());
  MOCK_METHOD1(updateImportanceSampler, void(const Eigen::Ref<const control_trajectory>& nominal_control));
  MOCK_METHOD0(allocateCUDAMemory, void());
};
#endif //MPPIGENERIC_MOCK_CONTROLLER_H
