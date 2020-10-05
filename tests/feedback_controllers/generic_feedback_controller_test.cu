#include <gtest/gtest.h>
#include <mppi_test/mock_classes/mock_dynamics.h>
#include <mppi/feedback_controllers/feedback.cuh>

class TestGPUFeedbackController : GPUFeedbackController<TestGPUFeedbackController, MockDynamics> {

};

class TestFeedbackController : FeedbackController<> {

};
