//
// Created by jgibson37 on 2/24/20.
//


#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <numeric>
#include <boost/thread.hpp>

#include <mppi/core/base_plant.hpp>
#include <mppi/instantiations/cartpole_mppi/cartpole_mppi.cuh>
#include <mppi_test/mock_classes/mock_dynamics.h>
#include <mppi_test/mock_classes/mock_controller.h>
#include <mppi_test/mock_classes/mock_costs.h>

template <class CONTROLLER_T>
class TestPlant : public BasePlant<CONTROLLER_T> {
public:
  double time_ = 0.0;

  double avgDurationMs_ = 0;
  double avgTickDuration_ = 0;
  double avgSleepTime_ = 0;

  using c_array = typename CONTROLLER_T::control_array;
  using c_traj = typename CONTROLLER_T::control_trajectory;

  using s_array = typename CONTROLLER_T::state_array;
  using s_traj = typename CONTROLLER_T::state_trajectory;
  using K_mat = typename CONTROLLER_T::K_matrix;

  using DYN_T = typename CONTROLLER_T::TEMPLATED_DYNAMICS;
  using DYN_PARAMS_T = typename DYN_T::DYN_PARAMS_T;
  using COST_T = typename CONTROLLER_T::TEMPLATED_COSTS;
  using COST_PARAMS_T = typename COST_T::COST_PARAMS_T;
  double timestamp_;
  double loop_speed_;

  TestPlant(std::shared_ptr<MockController> controller) : BasePlant<CONTROLLER_T>(controller,10,1) {}


  void pubControl(c_array& u) override {

  }

  void setTimingInfo(double avg_duration_ms, double avg_tick_duration, double avg_sleep_time) override {
    avgDurationMs_ = avg_duration_ms;
    avgTickDuration_ = avg_tick_duration;
    avgSleepTime_ = avg_sleep_time;
  }

  int checkStatus() override {
    return 1;
  }

  // accessors for protected members
  int getNumIter() {return this->num_iter_;}
};

typedef TestPlant<MockController> MockTestPlant;

TEST(BasePlant, Constructor) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);

  EXPECT_EQ(plant.hasNewCostParams(), false);
  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  EXPECT_EQ(plant.hasNewModel(), false);
  EXPECT_EQ(plant.hasNewCostmap(), false);
  EXPECT_EQ(plant.hasNewObstacles(), false);

  EXPECT_EQ(plant.getNumIter(), 0);
}

TEST(BasePlant, getAndSetState) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);
  // check initial state is zerod

  MockController::state_array state = plant.getState();
  for(int i = 0; i < 1; i++) {
    EXPECT_EQ(state(i), 0.0);
  }

  MockController::state_array new_state;
  for(int i = 0; i < 1; i++) {
    new_state(i) = i;
  }
  plant.setState(new_state);
  state = plant.getState();
  for(int i = 0; i < 1; i++) {
    EXPECT_EQ(state(i), i);
  }
}

TEST(BasePlant, getSetOptimizationStride) {

  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);
  int optimization_stride = plant.getOptimizationStride();

  EXPECT_EQ(optimization_stride, 1);

  plant.setOptimizationStride(5);
  optimization_stride = plant.getOptimizationStride();

  EXPECT_EQ(optimization_stride, 5);
}

TEST(BasePlant, getSetDynamicsParams) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);

  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  MockTestPlant::DYN_PARAMS_T params;

  params.test = 3;

  plant.setDynamicsParams(params);
  EXPECT_EQ(plant.hasNewDynamicsParams(), true);

  MockTestPlant::DYN_PARAMS_T new_params = plant.getNewDynamicsParams();
  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  EXPECT_EQ(new_params.test, params.test);
}

TEST(BasePlant, getSetCostParams) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);
  EXPECT_EQ(plant.hasNewCostParams(), false);

  MockTestPlant::COST_PARAMS_T params;
  params.test = 100;

  plant.setCostParams(params);
  EXPECT_EQ(plant.hasNewCostParams(), true);

  auto new_params = plant.getNewCostParams();
  EXPECT_EQ(plant.hasNewCostParams(), false);
  EXPECT_EQ(params.test, new_params.test);
}

TEST(BasePlant, runControlIterationStoppedTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(0);

  std::atomic<bool> is_alive(false);
  testPlant.runControlIteration(mockController.get(), &is_alive);
}
/*
 * TODO
TEST(BasePlant, runControlIterationDebugFalseTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(*mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(*mockController, getFeedbackGains()).Times(0);
  EXPECT_CALL(*mockController, getControlSeq()).Times(1);
  EXPECT_CALL(*mockController, getStateSeq()).Times(1);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(mockController.get(), &is_alive);

  EXPECT_EQ(testPlant.checkStatus(), 1);
  // check state
  // check control
  // check feedback gains
  // check last pose update
  // get avg optimization time
  // TODO check solution
}

TEST(BasePlant, runControlIterationDebugFalseUpdateAllTest) {

  // TODO look into mocking the dynamics and controller class
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);
  std::vector<int> map_desc;
  std::vector<float4> values;

  testPlant.setCostParams(mockCostParams());
  testPlant.setDynamicsParams(mockDynamicsParams());

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(1);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(*mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(*mockController, getControlSeq()).Times(1);
  EXPECT_CALL(*mockController, getStateSeq()).Times(1);
  EXPECT_CALL(*mockController, getFeedbackGains()).Times(0);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(mockController.get(), &is_alive);

  EXPECT_EQ(testPlant.checkStatus(), 1);
  // TODO check solution
}
 */

TEST(BasePlant, runControlIterationDebugNoDisplayTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);
  testPlant.setDebugMode(true);

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(1);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(*mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(*mockController, getControlSeq()).Times(1);
  EXPECT_CALL(*mockController, getStateSeq()).Times(1);
  EXPECT_CALL(*mockController, getFeedbackGains()).Times(0);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(mockController.get(), &is_alive);
}


TEST(BasePlant, runControlIterationDebugDisplayTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);
  testPlant.setDebugMode(true);

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(1).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(mockCost, getDebugDisplay(testing::_)).Times(1);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(*mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(*mockController, getControlSeq()).Times(1);
  EXPECT_CALL(*mockController, getStateSeq()).Times(1);
  EXPECT_CALL(*mockController, getFeedbackGains()).Times(0);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(mockController.get(), &is_alive);
}

/*
 * TODO getting last time is broken for this test, always the same
TEST(BasePlant, runControlLoop) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);
  int hz = testPlant.getHz();
  double time = 1; // in seconds

  // setup mock expected calls
  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(*mockController, resetControls()).Times(1);
  EXPECT_CALL(*mockController, slideControlSequence(1)).Times(hz/time);
  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(hz/time);
  EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(*mockController, getFeedbackGains()).Times(0);
  EXPECT_CALL(*mockController, getControlSeq()).Times(hz/time);
  EXPECT_CALL(*mockController, getStateSeq()).Times(hz/time);

  std::atomic<bool> is_alive(true);
  boost::thread optimizer;
  optimizer = boost::thread(boost::bind(&MockTestPlant::runControlLoop, &testPlant, mockController.get(), &is_alive));

  std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> loop_duration = std::chrono::steady_clock::now() - loop_start;
  while(loop_duration.count() < time*1e3) {
    usleep(50);
    loop_duration = std::chrono::steady_clock::now() - loop_start;
  }
  is_alive.store(false);
  optimizer.join();


  EXPECT_EQ(testPlant.checkStatus(), 1);
}
*/
