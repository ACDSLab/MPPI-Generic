//
// Created by jason on 2/21/24.
//

#ifndef MPPIGENERIC_TESTS_INCLUDE_MPPI_TEST_MOCK_CLASSES_MOCK_FEEDBACK_H_
#define MPPIGENERIC_TESTS_INCLUDE_MPPI_TEST_MOCK_CLASSES_MOCK_FEEDBACK_H_

#include <mppi_test/mock_classes/mock_controller.h>
#include <mppi/feedback_controllers/feedback.cuh>

struct mockGPUFeedbackParams
{
};

class MockGPUFeedback : public GPUFeedbackController<MockGPUFeedback, MockDynamics>
{
};

class MockFeedback : public FeedbackController<MockGPUFeedback, mockGPUFeedbackParams, NUM_TIMESTEPS>
{
};

#endif  // MPPIGENERIC_TESTS_INCLUDE_MPPI_TEST_MOCK_CLASSES_MOCK_FEEDBACK_H_
