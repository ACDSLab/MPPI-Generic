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
  using K_mat = typename CONTROLLER_T::feedback_gain_trajectory;

  using DYN_T = typename CONTROLLER_T::TEMPLATED_DYNAMICS;
  using DYN_PARAMS_T = typename DYN_T::DYN_PARAMS_T;
  using COST_T = typename CONTROLLER_T::TEMPLATED_COSTS;
  using COST_PARAMS_T = typename COST_T::COST_PARAMS_T;
  double timestamp_;
  double loop_speed_;

  TestPlant(std::shared_ptr<MockController> controller, int hz = 20, int opt_stride=1)
        : BasePlant<CONTROLLER_T>(controller,hz,opt_stride) {}


  void pubControl(const c_array& u) override {

  }

  void incrementTime() {
    time_ += 0.05;
  }

  void setTimingInfo(double avg_duration_ms, double avg_tick_duration, double avg_sleep_time) override {
    avgDurationMs_ = avg_duration_ms;
    avgTickDuration_ = avg_tick_duration;
    avgSleepTime_ = avg_sleep_time;
  }

  int checkStatus() override {
    return 1;
  }

  double getCurrentTime() {
    return time_;
  }

  // accessors for protected members
  int getNumIter() {return this->num_iter_;}
  double getLastUsedPoseUpdateTime() {return this->last_used_pose_update_time_;}
  int getStatus() {return this->status_;}
  bool getDebugMode() {return this->debug_mode_;}
  double getOptimizationDuration() {return this->optimization_duration_;}
  double getOptimizationAvg() {return this->avg_optimize_time_ms_;}
  double getLoopDuration() {return this->optimize_loop_duration_;}
  double getLoopAvg() {return this->avg_loop_time_ms_;}
  double getFeedbackDuration() {return this->feedback_duration_;}
  double getFeedbackAvg() {return this->avg_feedback_time_ms_;}
  void setLastTime(double time) {this->last_used_pose_update_time_ = time;}
};

typedef TestPlant<MockController> MockTestPlant;

TEST(BasePlant, Constructor) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);

  EXPECT_EQ(plant.controller_, mockController);
  EXPECT_EQ(plant.getHz(), 20);
  EXPECT_EQ(plant.getTargetOptimizationStride(), 1);
  EXPECT_EQ(plant.getNumIter(), 0);
  EXPECT_EQ(plant.getLastUsedPoseUpdateTime(), -1);
  EXPECT_EQ(plant.getStatus(), 1);

  EXPECT_EQ(plant.hasNewCostParams(), false);
  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  EXPECT_EQ(plant.hasNewModel(), false);
  EXPECT_EQ(plant.hasNewCostmap(), false);
  EXPECT_EQ(plant.hasNewObstacles(), false);
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
  int optimization_stride = plant.getTargetOptimizationStride();

  EXPECT_EQ(optimization_stride, 1);

  plant.setTargetOptimizationStride(5);
  optimization_stride = plant.getTargetOptimizationStride();

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

TEST(BasePlant, updateParametersAllFalse) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, getDebugDisplay(testing::_)).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockCost, updateCostmap(testing::_, testing::_)).Times(0);

  MockDynamics::state_array state = MockDynamics::state_array::Zero();
  testPlant.updateParameters(mockController.get(), state);
}


TEST(BasePlant, updateParametersAllTrue) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(1).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(mockCost, getDebugDisplay(testing::_)).Times(1);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(1);
  // TODO implement updating costmap
  EXPECT_CALL(mockCost, updateCostmap(testing::_, testing::_)).Times(0);

  testPlant.setDebugMode(true);
  testPlant.setDynamicsParams(MockDynamics::DYN_PARAMS_T());
  testPlant.setCostParams(MockCost::COST_PARAMS_T());

  MockDynamics::state_array state = MockDynamics::state_array::Zero();
  testPlant.updateParameters(mockController.get(), state);
}

TEST(BasePlant, updateStateOutsideTimeTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  mockController->setDt(0.05);

  MockTestPlant testPlant(mockController);
  testPlant.setLastTime(0);

  EXPECT_CALL(*mockController, getCurrentControl(testing::_, testing::_)).Times(0);

  MockController::state_array state = MockController::state_array::Zero();
  testPlant.updateState(state, mockController->getDt() * mockController->getNumTimesteps() + 0.01);
  EXPECT_EQ(testPlant.getState(), state);

  testPlant.setLastTime(100);
  testPlant.updateState(state, 99.99);
  EXPECT_EQ(testPlant.getState(), state);
}


TEST(BasePlant, updateStateTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  mockController->setDt(0.05);

  MockTestPlant testPlant(mockController);
  testPlant.setLastTime(0);

  MockController::state_array state = MockController::state_array::Zero();
  EXPECT_CALL(*mockController, getCurrentControl(state, mockController->getDt())).Times(1);
  testPlant.updateState(state, mockController->getDt());
  EXPECT_EQ(testPlant.getState(), state);

  //EXPECT_CALL(*mockController, getCurrentControl(state, mockController->getDt()+100)).Times(1);
  //testPlant.setLastTime(100);
  //testPlant.updateState(state, 100+mockController->getDt());
  //EXPECT_EQ(testPlant.getState(), state);
}

TEST(BasePlant, runControlIterationStoppedTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  EXPECT_CALL(*mockController, slideControlSequence(testing::_)).Times(0);
  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(0);

  std::atomic<bool> is_alive(false);
  testPlant.runControlIteration(mockController.get(), &is_alive);
}

// TODO speed up to make tests run faster
TEST(BasePlant, runControlIterationDebugFalseNoFeedbackTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  for(int i = 0; i < 2; i++) {
    double wait_ms = 50*i;

    auto wait_function = [wait_ms]() {
      usleep(wait_ms*1e3);
    };

    int expect_opt_stride = i > 0 ? 1 : 0;

    EXPECT_CALL(*mockController, slideControlSequence(expect_opt_stride)).Times(i > 0 ? 1 : 0);
    EXPECT_CALL(*mockController, computeControl(testing::_)).Times(1).WillRepeatedly(testing::Invoke(wait_function));
    MockController::control_trajectory control_seq = MockController::control_trajectory::Zero();
    EXPECT_CALL(*mockController, getControlSeq()).Times(1).WillRepeatedly(testing::Return(control_seq));
    MockController::state_trajectory state_seq = MockController::state_trajectory::Zero();
    EXPECT_CALL(*mockController, getStateSeq()).Times(1).WillRepeatedly(testing::Return(state_seq));

    EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(0);
    EXPECT_CALL(*mockController, getFeedbackGains()).Times(0);

    EXPECT_EQ(testPlant.getDebugMode(), false);

    std::atomic<bool> is_alive(true);
    testPlant.runControlIteration(mockController.get(), &is_alive);
    testPlant.incrementTime();

    EXPECT_EQ(testPlant.checkStatus(), 1);
    EXPECT_EQ(testPlant.getStateTraj(), state_seq);
    EXPECT_EQ(testPlant.getControlTraj(), control_seq);
    MockController::feedback_gain_trajectory feedback = testPlant.getFeedbackGains();
    MockController::state_array state = MockController::state_array::Ones();
    for(int j = 0; j < 100; j++) {
      // TODO check that feedback is correct
      //auto result = feedback[i] * state;
      //float sum = feedback[i] * state;
      //EXPECT_FLOAT_EQ(result(0), 0);
    }

    // check last pose update
    EXPECT_FLOAT_EQ(testPlant.getLastUsedPoseUpdateTime(), 0.05*i);
    EXPECT_EQ(testPlant.getNumIter(), i+1);
    EXPECT_EQ(testPlant.getLastOptimizationStride(), expect_opt_stride);

    double small_time_ms = 2; // how long we should expect a non delayed call to take
    EXPECT_THAT(testPlant.getOptimizationDuration(),
            testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
    EXPECT_LT(testPlant.getOptimizationAvg(), wait_ms + small_time_ms);
    EXPECT_THAT(testPlant.getLoopDuration(),
                testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
    EXPECT_LT(testPlant.getLoopAvg(), wait_ms + small_time_ms);
    EXPECT_LE(testPlant.getFeedbackDuration(), small_time_ms);
    EXPECT_LE(testPlant.getFeedbackAvg(), small_time_ms);
  }

}

TEST(BasePlant, runControlIterationDebugFalseFeedbackTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  mockController->setFeedbackController(true);
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  for(int i = 0; i < 10; i++) {
    double wait_ms = 50*i;

    auto wait_function = [wait_ms]() {
      usleep(wait_ms*1e3);
    };

    int expect_opt_stride = i > 0 ? 1 : 0;

    EXPECT_CALL(*mockController, slideControlSequence(expect_opt_stride)).Times(i > 0 ? 1 : 0);
    EXPECT_CALL(*mockController, computeControl(testing::_)).Times(1).WillRepeatedly(testing::Invoke(wait_function));
    MockController::control_trajectory control_seq = MockController::control_trajectory::Zero();
    EXPECT_CALL(*mockController, getControlSeq()).Times(1).WillRepeatedly(testing::Return(control_seq));
    MockController::state_trajectory state_seq = MockController::state_trajectory::Zero();
    EXPECT_CALL(*mockController, getStateSeq()).Times(1).WillRepeatedly(testing::Return(state_seq));

    EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(1).WillRepeatedly(testing::Invoke(wait_function));
    MockController::feedback_gain_trajectory feedback;
    EXPECT_CALL(*mockController, getFeedbackGains()).Times(1).WillRepeatedly(testing::Return(feedback));

    EXPECT_EQ(testPlant.getDebugMode(), false);

    std::atomic<bool> is_alive(true);
    testPlant.runControlIteration(mockController.get(), &is_alive);
    testPlant.incrementTime();

    EXPECT_EQ(testPlant.checkStatus(), 1);
    EXPECT_EQ(testPlant.getStateTraj(), state_seq);
    EXPECT_EQ(testPlant.getControlTraj(), control_seq);
    EXPECT_EQ(testPlant.getFeedbackGains(), feedback);

    // check last pose update
    EXPECT_FLOAT_EQ(testPlant.getLastUsedPoseUpdateTime(), 0.05*i);
    EXPECT_EQ(testPlant.getNumIter(), i+1);
    EXPECT_EQ(testPlant.getLastOptimizationStride(), expect_opt_stride);

    double small_time_ms = 10; // how long we should expect a non delayed call to take
    EXPECT_THAT(testPlant.getOptimizationDuration(),
                testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
    EXPECT_LT(testPlant.getOptimizationAvg(), wait_ms + small_time_ms);
    EXPECT_THAT(testPlant.getFeedbackDuration(),
                testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
    // TODO should be range as well
    EXPECT_LT(testPlant.getFeedbackAvg(), wait_ms + small_time_ms);
    EXPECT_THAT(testPlant.getLoopDuration(),
                testing::AllOf(testing::Ge(wait_ms*2), testing::Le((wait_ms + small_time_ms)*2)));
    EXPECT_LT(testPlant.getLoopAvg(), (wait_ms + small_time_ms)*2);
  }

}

TEST(BasePlant, runControlIterationDebugFalseFeedbackAvgTest) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  mockController->setFeedbackController(true);
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);

  for(int i = 0; i < 10; i++) {
    double wait_ms = 50;

    auto wait_function = [wait_ms]() {
      usleep(wait_ms*1e3);
    };

    int expect_opt_stride = i > 0 ? 1 : 0;

    EXPECT_CALL(*mockController, slideControlSequence(expect_opt_stride)).Times(i > 0 ? 1 : 0);
    EXPECT_CALL(*mockController, computeControl(testing::_)).Times(1).WillRepeatedly(testing::Invoke(wait_function));
    MockController::control_trajectory control_seq = MockController::control_trajectory::Zero();
    EXPECT_CALL(*mockController, getControlSeq()).Times(1).WillRepeatedly(testing::Return(control_seq));
    MockController::state_trajectory state_seq = MockController::state_trajectory::Zero();
    EXPECT_CALL(*mockController, getStateSeq()).Times(1).WillRepeatedly(testing::Return(state_seq));

    EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(1).WillRepeatedly(testing::Invoke(wait_function));
    MockController::feedback_gain_trajectory feedback;
    EXPECT_CALL(*mockController, getFeedbackGains()).Times(1).WillRepeatedly(testing::Return(feedback));

    EXPECT_EQ(testPlant.getDebugMode(), false);

    std::atomic<bool> is_alive(true);
    testPlant.runControlIteration(mockController.get(), &is_alive);
    testPlant.incrementTime();

    EXPECT_EQ(testPlant.checkStatus(), 1);
    EXPECT_EQ(testPlant.getStateTraj(), state_seq);
    EXPECT_EQ(testPlant.getControlTraj(), control_seq);
    EXPECT_EQ(testPlant.getFeedbackGains(), feedback);

    // check last pose update
    EXPECT_FLOAT_EQ(testPlant.getLastUsedPoseUpdateTime(), 0.05*i);
    EXPECT_EQ(testPlant.getNumIter(), i+1);
    EXPECT_EQ(testPlant.getLastOptimizationStride(), expect_opt_stride);

    double small_time_ms = 10; // how long we should expect a non delayed call to take
    EXPECT_THAT(testPlant.getOptimizationDuration(),
                testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
    EXPECT_THAT(testPlant.getOptimizationAvg(),
                testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
    EXPECT_THAT(testPlant.getLoopDuration(),
                testing::AllOf(testing::Ge(wait_ms*2), testing::Le((wait_ms + small_time_ms)*2)));
    EXPECT_THAT(testPlant.getLoopAvg(),
                testing::AllOf(testing::Ge((wait_ms)*2), testing::Le((wait_ms + small_time_ms)*2)));
    EXPECT_THAT(testPlant.getFeedbackDuration(),
                testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
    EXPECT_THAT(testPlant.getFeedbackAvg(),
                testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
  }

}

TEST(BasePlant, runControlLoop) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  mockController->setFeedbackController(true);
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant testPlant(mockController);
  int hz = testPlant.getHz();
  double time = 1.0; // in seconds

  // setup mock expected calls
  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  EXPECT_CALL(*mockController, resetControls()).Times(1);

  double wait_s = (1.0/hz)/2; // divide by 2 since wait is evenly split across computeFeedbackGains and computeControl

  auto wait_function = [wait_s]() {
    usleep(wait_s*1e6);
  };
  int iterations = int(std::round((hz*1.0) / (time * 1.0))); // number of times the method will be called
  EXPECT_CALL(*mockController, slideControlSequence(1)).Times(iterations/2);
  EXPECT_CALL(*mockController, computeControl(testing::_)).Times(iterations/2).WillRepeatedly(testing::Invoke(wait_function));
  MockController::control_trajectory control_seq = MockController::control_trajectory::Zero();
  EXPECT_CALL(*mockController, getControlSeq()).Times(iterations/2).WillRepeatedly(testing::Return(control_seq));
  MockController::state_trajectory state_seq = MockController::state_trajectory::Zero();
  EXPECT_CALL(*mockController, getStateSeq()).Times(iterations/2).WillRepeatedly(testing::Return(state_seq));
  EXPECT_CALL(*mockController, computeFeedbackGains(testing::_)).Times(iterations/2).WillRepeatedly(testing::Invoke(wait_function));
  MockController::feedback_gain_trajectory feedback;
  EXPECT_CALL(*mockController, getFeedbackGains()).Times(iterations/2).WillRepeatedly(testing::Return(feedback));

  std::atomic<bool> is_alive(true);
  boost::thread optimizer;
  optimizer = boost::thread(boost::bind(&MockTestPlant::runControlLoop, &testPlant, mockController.get(), &is_alive));

  std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> loop_duration = std::chrono::steady_clock::now() - loop_start;
  int counter = 0;
  while(loop_duration.count() < time*1e3) {
    counter++;
    while(loop_duration.count() < (time/hz)*1e3*counter) {
      usleep(50);
      loop_duration = std::chrono::steady_clock::now() - loop_start;
    }
    if(counter >= iterations / 2) { // this forces it to block
      testPlant.incrementTime();
    }
  }
  is_alive.store(false);
  optimizer.join();

  // check all the things
  EXPECT_EQ(testPlant.checkStatus(), 1);

  EXPECT_EQ(testPlant.checkStatus(), 1);
  EXPECT_EQ(testPlant.getStateTraj(), state_seq);
  EXPECT_EQ(testPlant.getControlTraj(), control_seq);
  EXPECT_EQ(testPlant.getFeedbackGains(), feedback);

  // check last pose update
  EXPECT_NE(testPlant.getLastUsedPoseUpdateTime(), 0.0);
  EXPECT_EQ(testPlant.getNumIter(), iterations/2);
  EXPECT_EQ(testPlant.getLastOptimizationStride(), 1);

  double small_time_ms = 10; // how long we should expect a non delayed call to take
  double wait_ms = wait_s*1e3;
  EXPECT_THAT(testPlant.getOptimizationDuration(),
              testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
  EXPECT_THAT(testPlant.getOptimizationAvg(),
              testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
  EXPECT_THAT(testPlant.getLoopDuration(),
              testing::AllOf(testing::Ge(wait_ms*2), testing::Le((wait_ms + small_time_ms)*2)));
  EXPECT_THAT(testPlant.getLoopAvg(),
              testing::AllOf(testing::Ge((wait_ms)*2), testing::Le((wait_ms + small_time_ms)*2)));
  EXPECT_THAT(testPlant.getFeedbackDuration(),
              testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
  EXPECT_THAT(testPlant.getFeedbackAvg(),
              testing::AllOf(testing::Ge(wait_ms), testing::Le(wait_ms + small_time_ms)));
}
