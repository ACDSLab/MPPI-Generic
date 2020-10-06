#include <gtest/gtest.h>
#include <mppi_test/mock_classes/mock_dynamics.h>
#include <mppi/feedback_controllers/feedback.cuh>
struct DynamicsTesterParams {
  int var_1 = 1;
  int var_2 = 2;
  float4 var_4;
};

template<int STATE_DIM = 1, int CONTROL_DIM = 1>
class DynamicsTester : public MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM> {
public:

  using state_array = typename MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>::state_array;
  using control_array = typename MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>::control_array;

  DynamicsTester(cudaStream_t stream=0)
          : MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>(stream) {}

  DynamicsTester(std::array<float2, CONTROL_DIM> control_rngs, cudaStream_t stream=0)
          : MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>(control_rngs, stream) {}

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der) {
    state_der(1) = control(0);
  }

  void computeKinematics(const Eigen::Ref<const state_array> &state, Eigen::Ref<state_array> s_der) {
    s_der(0) = state(0) + state(1);
  };

  // TODO must be properly parallelized
  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s = nullptr) {
    state_der[1] = control[0];
  }

  // TODO must be properly parallelized
  __device__ void computeKinematics(float* state, float* state_der) {
    state_der[0] = state[0] + state[1];
  }
};

class TestGPUFeedbackController : public GPUFeedbackController<TestGPUFeedbackController, DynamicsTester<>> {
public:
  typedef MockDynamics DYN_T;

  TestGPUFeedbackController(cudaStream_t stream = 0) : GPUFeedbackController<TestGPUFeedbackController, DynamicsTester<>>(stream) {}

  void allocateCudaMemory() {

  }

  void deallocateCUDAMemory() {

  }

  void copyToDevice() {

  }

  void copyFromDevice() {

  }

};

class TestFeedbackController : public FeedbackController<TestGPUFeedbackController, 10> {
public:
  TestFeedbackController(cudaStream_t stream = 0) : FeedbackController<TestGPUFeedbackController, 10>(stream) {}

  std::shared_ptr<TestGPUFeedbackController> getGPUPointer() {return this->gpu_controller_;}

  control_array k(const Eigen::Ref<state_array>& x_act,
                          const Eigen::Ref<state_array>& x_goal, float t) override {

  }

  // might not be a needed method
  void computeFeedbackGains(const Eigen::Ref<const state_array>& init_state,
                                    const Eigen::Ref<const state_trajectory>& goal_traj,
                                    const Eigen::Ref<const control_trajectory>& control_traj) override {

  }
};

TEST(FeedbackController, Constructor) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  TestFeedbackController feedbackController(stream);

  EXPECT_EQ(feedbackController.getGPUPointer()->stream_, stream) << "Stream binding failure.";
  EXPECT_EQ(feedbackController.getGPUPointer()->GPUMemStatus_, true);
  EXPECT_NE(feedbackController.getDevicePointer(), nullptr);

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(FeedbackController, ) {

}
