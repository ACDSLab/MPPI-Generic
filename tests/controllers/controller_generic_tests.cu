#include <mppi/controllers/controller.cuh>
#include <mppi_test/mock_classes/mock_costs.h>
#include <mppi_test/mock_classes/mock_dynamics.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <numeric>


static const int number_rollouts = 1200;

class TestController : public Controller<MockDynamics, MockCost, 100, number_rollouts, 1, 2>{
public:
  TestController(MockDynamics* model, MockCost* cost, float dt, int max_iter, float gamma,
                 const Eigen::Ref<const control_array>& control_variance,
                 int num_timesteps = 100,
                 const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                 cudaStream_t stream = nullptr) : Controller<MockDynamics, MockCost, 100, number_rollouts, 1, 2>(
                         model, cost, dt, max_iter, gamma, control_variance, num_timesteps,
                         init_control_traj, stream) {

    // Allocate CUDA memory for the controller
    allocateCUDAMemoryHelper(0);

    // Copy the noise variance to the device
    this->copyControlStdDevToDevice();
  }

  virtual void computeControl(const Eigen::Ref<const state_array>& state) override {

  }

  void computeControl(const Eigen::Ref<const state_array>& state,
                      const std::array<control_trajectory, number_rollouts> noise) {
    int trajectory_size = control_trajectory().size();
    for (int i = 0; i < number_rollouts; i++) {
      HANDLE_ERROR(cudaMemcpyAsync(control_noise_d_ + i * trajectory_size,
                                   noise[i].data(),
                                   sizeof(float)*trajectory_size,
                                   cudaMemcpyHostToDevice, stream_));
    }
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
    // Normally rolloutKernel would be called here and would transform
    //  control_noise_d_ from u to u + noise

    // Instead we just get back noise in this test
    this->copySampledControlFromDevice();
  }

  virtual void slideControlSequence(int steps) override {

  }

  float getDt() {return dt_;}
  int getNumIter() {return num_iters_;}
  float getGamma() {return gamma_;}
  float getNumTimesteps() {return num_timesteps_;}
  cudaStream_t getStream() {return stream_;}

  void setFeedbackGains(TestController::feedback_gain_trajectory traj) {
    this->result_.feedback_gain = traj;
  }
};

TEST(Controller, ConstructorDestructor) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);
  int num_timesteps = 10;
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  TestController::control_trajectory init_control_trajectory = TestController::control_trajectory::Ones();

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(stream)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(stream)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController* controller = new TestController(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var, num_timesteps,
          init_control_trajectory, stream);

  EXPECT_EQ(controller->model_, &mockDynamics);
  EXPECT_EQ(controller->cost_, &mockCost);
  EXPECT_EQ(controller->getDt(), dt);
  EXPECT_EQ(controller->getNumIter(), max_iter);
  EXPECT_EQ(controller->getGamma(), gamma);
  EXPECT_EQ(controller->getNumTimesteps(), num_timesteps);
  EXPECT_EQ(controller->getControlStdDev(), control_var);
  EXPECT_EQ(controller->getControlSeq(), init_control_trajectory);
  EXPECT_EQ(controller->getStream(), stream);
  EXPECT_EQ(controller->getFeedbackEnabled(), false);

  // TODO check that a random seed was set and stream was set
  //EXPECT_NE(controller->getRandomSeed(), 0);

  // TODO check for correct defaults
  delete controller;
}

TEST(Controller, setNumTimesteps) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);

  controller.setNumTimesteps(10);
  EXPECT_EQ(controller.getNumTimesteps(), 10);

  controller.setNumTimesteps(1000);
  EXPECT_EQ(controller.getNumTimesteps(), 100);
}


TEST(Controller, updateControlNoiseStdDev) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);

  TestController::control_array new_control_var = TestController::control_array::Ones();

  controller.updateControlNoiseStdDev(new_control_var);

  EXPECT_EQ(controller.getControlStdDev(), new_control_var);
  // TODO verify copied to GPU correctly
}

TEST(Controller, slideControlSequenceHelper) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);
  TestController::control_trajectory u;
  for(int i = 0; i < controller.num_timesteps_; i++) {
    TestController::control_array control = TestController::control_array::Ones();
    control = control * i;
    u.col(i) = control.transpose();
  }

  controller.slideControlSequenceHelper(1, u);
  for(int i = 0; i < controller.num_timesteps_; i++) {
    for(int j = 0; j < MockDynamics::CONTROL_DIM; j++) {
      int val = std::min(i + 1, controller.num_timesteps_ - 1);
      EXPECT_FLOAT_EQ(u(j, i), val);
    }
  }

  controller.slideControlSequenceHelper(10, u);
  for(int i = 0; i < controller.num_timesteps_; i++) {
    for(int j = 0; j < MockDynamics::CONTROL_DIM; j++) {
      int val = std::min(i + 11, controller.num_timesteps_ - 1);
      EXPECT_FLOAT_EQ(u(j, i), val);
    }
  }
}

TEST(Controller, computeStateTrajectoryHelper) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);

  TestController::state_array x = TestController::state_array::Ones();
  TestController::state_array xdot = TestController::state_array::Ones();
  EXPECT_CALL(mockDynamics, computeStateDeriv(testing::_, testing::_, testing::_)).Times(controller.num_timesteps_ - 1);
  EXPECT_CALL(mockDynamics, updateState(testing::_, testing::_, dt)).Times(controller.num_timesteps_ - 1);
  
  TestController::state_trajectory result = TestController::state_trajectory::Ones();
  TestController::control_trajectory u = TestController::control_trajectory::Zero();
  controller.computeStateTrajectoryHelper(result, x, u);

  for(int i = 0; i < controller.num_timesteps_; i++) {
    for(int j = 0; j < MockDynamics::STATE_DIM; j++){
      EXPECT_FLOAT_EQ(result(j, i), 1.0);
    }
  }
}

TEST(Controller, interpolateControl) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);
  TestController::control_trajectory traj;
  for(int i = 0; i < controller.getNumTimesteps(); i++) {
    traj.col(i) = TestController::control_array::Ones() * i;
  }
  controller.updateImportanceSampler(traj);

  for(double i = 0; i < controller.getNumTimesteps() - 1; i+= 0.25) {
    TestController::control_array result = controller.interpolateControls(i*controller.getDt());
    EXPECT_FLOAT_EQ(result(0), i) << i;
  }
}

TEST(Controller, interpolateFeedback) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);

  controller.setFeedbackController(true);
  TestController::feedback_gain_trajectory feedback_traj = TestController::feedback_gain_trajectory(controller.getNumTimesteps());
  for(int i = 0; i < controller.getNumTimesteps(); i++) {
    feedback_traj[i] = Eigen::Matrix<float, 1, 1>::Ones() * i;
  }
  controller.setFeedbackGains(feedback_traj);

  TestController::state_array state = TestController::state_array::Ones();
  for(double i = 0; i < controller.getNumTimesteps() - 1; i += 0.25) {
    TestController::control_array result = controller.interpolateFeedback(state, i*controller.getDt());
    EXPECT_FLOAT_EQ(result(0), i);
  }
}


TEST(Controller, getCurrentControlTest) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);

  EXPECT_CALL(mockDynamics, enforceConstraints(testing::_, testing::_)).Times(4 * (controller.getNumTimesteps() - 1));

  controller.setFeedbackController(true);
  TestController::feedback_gain_trajectory feedback_traj = TestController::feedback_gain_trajectory(controller.getNumTimesteps());
  TestController::control_trajectory traj;
  for(int i = 0; i < controller.getNumTimesteps(); i++) {
    feedback_traj[i] = Eigen::Matrix<float, 1, 1>::Ones() * i;
    traj.col(i) = TestController::control_array::Ones() * i;
  }
  controller.setFeedbackGains(feedback_traj);
  controller.updateImportanceSampler(traj);

  TestController::state_array state = TestController::state_array::Ones();
  for(double i = 0; i < controller.getNumTimesteps() - 1; i += 0.25) {
    TestController::control_array result = controller.getCurrentControl(state, i*controller.getDt());
    EXPECT_FLOAT_EQ(result(0), i*2);
  }
}

TEST(Controller, getSampledControlTrajectories) {
  // Create controller
  // Use computeControl with noise passed in
  // Inside computeControl copySampledControlFromDevice is used
  // Get sampled control sequence
  // Compare to original noise
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_var;
  control_var = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_var);

  // Create noisy trajectories./
  std::array<TestController::control_trajectory, number_rollouts> noise;
  for(int i = 0; i < number_rollouts; i++) {
    noise[i] = TestController::control_trajectory::Random();
  }
  // Save back a percentage of trajectories
  controller.setPercentageSampledControlTrajectories(0.3);

  TestController::state_array x = TestController::state_array::Ones();
  controller.computeControl(x, noise);
  std::vector<TestController::control_trajectory> sampled_controls = controller.getSampledControlSeq();
  int j;
  float total_difference;
  for (int i = 0; i < sampled_controls.size(); i++) {
    float diff = -1;
    // Need to find which noise trajectory the current sample matches
    for (j = 0; j < number_rollouts; j++){
      diff = std::abs((noise[j] - sampled_controls[i]).norm());
      if (diff == 0) {
        break;
      }
    }
    total_difference += diff;
  }
  EXPECT_FLOAT_EQ(0, total_difference);
}

TEST(Controller, saveControlHistoryHelper_1) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  int steps = 1;
  MockDynamics::control_array control_std_dev;
  control_std_dev = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_std_dev);

  TestController::control_trajectory u = TestController::control_trajectory::Random();
  Eigen::Matrix<float, MockDynamics::CONTROL_DIM, 2> u_history;
  u_history.setOnes();

  controller.saveControlHistoryHelper(steps, u, u_history);

  for (int i = 0; i < MockDynamics::CONTROL_DIM; ++i) {
    EXPECT_FLOAT_EQ(u_history(i, 0), 1.0f) << "History column 0 failed";
    EXPECT_FLOAT_EQ(u_history(i, 1), u(i, steps-1)) << "History column 1 failed";
  }
}

TEST(Controller, saveControlHistoryHelper_2) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  int steps = 4;
  MockDynamics::control_array control_std_dev;
  control_std_dev = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_std_dev);

  TestController::control_trajectory u = TestController::control_trajectory::Random();
  Eigen::Matrix<float, MockDynamics::CONTROL_DIM, 2> u_history;
  u_history.setOnes();

  controller.saveControlHistoryHelper(steps, u, u_history);

  for (int i = 0; i < MockDynamics::CONTROL_DIM; ++i) {
    EXPECT_FLOAT_EQ(u_history(i, 0), u(i, steps-2)) << "History column 0 failed";
    EXPECT_FLOAT_EQ(u_history(i, 1), u(i, steps-1)) << "History column 1 failed";
  }
}
