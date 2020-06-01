#include <mppi/controllers/controller.cuh>
#include <mppi_test/mock_classes/mock_costs.h>
#include <mppi_test/mock_classes/mock_dynamics.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <numeric>

class TestController : public Controller<MockDynamics, MockCost, 100, 1200, 1, 2>{
public:
  TestController(MockDynamics* model, MockCost* cost, float dt, int max_iter, float gamma,
                 const Eigen::Ref<const control_array>& control_std_dev,
                 int num_timesteps = 100,
                 const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                 cudaStream_t stream = nullptr) : Controller<MockDynamics, MockCost, 100, 1200, 1, 2>(
                         model, cost, dt, max_iter, gamma, control_std_dev, num_timesteps,
                         init_control_traj, stream) {

    // Allocate CUDA memory for the controller
    allocateCUDAMemoryHelper(0);

    // Copy the noise std_dev to the device
    this->copyControlStdDevToDevice();
  }

  virtual void computeControl(const Eigen::Ref<const state_array>& state) override {

  }

  virtual void slideControlSequence(int steps) override {

  }

  virtual void updateImportanceSampler(const Eigen::Ref<const control_trajectory>& nominal_control) override {

  }

  float getDt() {return dt_;}
  int getNumIter() {return num_iters_;}
  float getGamma() {return gamma_;}
  float getNumTimesteps() {return num_timesteps_;}
  cudaStream_t getStream() {return stream_;}
};

TEST(Controller, ConstructorDestructor) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_std_dev;
  control_std_dev = MockDynamics::control_array::Constant(1.0);
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

  TestController* controller = new TestController(&mockDynamics, &mockCost, dt, max_iter, gamma, control_std_dev, num_timesteps,
          init_control_trajectory, stream);

  EXPECT_EQ(controller->model_, &mockDynamics);
  EXPECT_EQ(controller->cost_, &mockCost);
  EXPECT_EQ(controller->getDt(), dt);
  EXPECT_EQ(controller->getNumIter(), max_iter);
  EXPECT_EQ(controller->getGamma(), gamma);
  EXPECT_EQ(controller->getNumTimesteps(), num_timesteps);
  EXPECT_EQ(controller->getControlStdDev(), control_std_dev);
  EXPECT_EQ(controller->getControlSeq(), init_control_trajectory);
  EXPECT_EQ(controller->getStream(), stream);

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
  MockDynamics::control_array control_std_dev;
  control_std_dev = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_std_dev);

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
  MockDynamics::control_array control_std_dev;
  control_std_dev = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_std_dev);

  TestController::control_array new_control_std_dev = TestController::control_array::Ones();

  controller.updateControlNoiseStdDev(new_control_std_dev);

  EXPECT_EQ(controller.getControlStdDev(), new_control_std_dev);
  // TODO verify copied to GPU correctly
}

TEST(Controller, slideControlSequenceHelper) {
  MockCost mockCost;
  MockDynamics mockDynamics;

  float dt = 0.1;
  int max_iter = 1;
  float gamma = 1.2;
  MockDynamics::control_array control_std_dev;
  control_std_dev = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_std_dev);
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
  MockDynamics::control_array control_std_dev;
  control_std_dev = MockDynamics::control_array::Constant(1.0);

  // expect double check rebind
  EXPECT_CALL(mockCost, bindToStream(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, bindToStream(testing::_)).Times(1);

  // expect GPU setup called again
  EXPECT_CALL(mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(mockDynamics, GPUSetup()).Times(1);

  TestController controller(&mockDynamics, &mockCost, dt, max_iter, gamma, control_std_dev);

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
