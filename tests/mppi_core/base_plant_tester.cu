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


// ===== mock cost ====
typedef struct {
  int test = 1;
} mockCostParams;

class MockCost : public Cost<MockCost, mockCostParams> {
public:
  MOCK_METHOD1(bindToStream, void(cudaStream_t stream));
  MOCK_METHOD0(getDebugDisplayEnabled, bool());
  MOCK_METHOD1(getDebugDisplay, cv::Mat(float* array));
  MOCK_METHOD1(setParams, void(mockCostParams params));
  MOCK_METHOD2(updateCostmap, void(std::vector<int> desc, std::vector<float> data));
};

// ===== mock dynamics ====
typedef struct {
  int test = 2;
} mockDynamicsParams;

class MockDynamics : public Dynamics<MockDynamics, mockDynamicsParams, 1, 1> {
public:
  MOCK_METHOD1(setParams, void(mockDynamicsParams params));
};

// ===== mock controller ====
class MockController : public Controller<MockDynamics, MockCost, 100, 500, 32, 2> {
public:
  MOCK_METHOD0(resetControls, void());
  MOCK_METHOD1(computeFeedbackGains, void(const Eigen::Ref<const state_array>& state));
  MOCK_METHOD1(slideControlSequence, void(int stride));
  MOCK_METHOD1(computeControl, void(const Eigen::Ref<const state_array>& state));
  MOCK_METHOD0(getControlSeq, control_trajectory());
  MOCK_METHOD0(getStateSeq, state_trajectory());
  MOCK_METHOD0(getFeedbackGains, K_matrix());
};

template <class CONTROLLER_T>
class TestPlant : public BasePlant<CONTROLLER_T> {
public:
  bool setDebugImage_ = false;
  double time_ = 0.0;

  double avgDurationMs_ = 0;
  double avgTickDuration_ = 0;
  double avgSleepTime_ = 0;

  typename CONTROLLER_T::state_trajectory state_seq_;
  typename CONTROLLER_T::control_trajectory control_seq_;
  typename CONTROLLER_T::K_matrix feedback_gains_;
  double timestamp_;
  double loop_speed_;

  TestPlant() : BasePlant<CONTROLLER_T>(10,1) {}


  double getLastPoseTime() override {
    double temp_time = time_;
    time_ += 0.5;
    return temp_time;
  }

  void setTimingInfo(double avg_duration_ms, double avg_tick_duration, double avg_sleep_time) override {
    avgDurationMs_ = avg_duration_ms;
    avgTickDuration_ = avg_tick_duration;
    avgSleepTime_ = avg_sleep_time;
  }

  void setDebugImage(cv::Mat debug_img) override {
    setDebugImage_ = true;
  }

  void setSolution(const typename CONTROLLER_T::state_trajectory& state_seq,
                           const typename CONTROLLER_T::control_trajectory& control_seq,
                           const typename CONTROLLER_T::K_matrix& feedback_gains,
                           double timestamp,
                           double loop_speed) override {
    state_seq_ = state_seq;
    control_seq_ = control_seq;
    feedback_gains_ = feedback_gains;
    timestamp_ = timestamp;
    loop_speed_ = loop_speed;
  }

  int checkStatus() override {
    return 1;
  }

  // accessors for protected members
  int getNumIter() {return this->num_iter_;}
};

typedef TestPlant<Controller<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>> CartpoleTestPlant;

TEST(BasePlant, Constructor) {
  CartpoleTestPlant plant;

  EXPECT_EQ(plant.hasNewCostParams(), false);
  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  EXPECT_EQ(plant.hasNewModel(), false);
  EXPECT_EQ(plant.hasNewCostmap(), false);
  EXPECT_EQ(plant.hasNewObstacles(), false);

  EXPECT_EQ(plant.getNumIter(), 0);
}

TEST(BasePlant, getAndSetState) {
  CartpoleTestPlant plant;
  CartpoleTestPlant::s_array state = plant.getState();
  // check initial state is zerod
  for(int i = 0; i < 4; i++) {
    EXPECT_EQ(state(i), 0.0);
  }

  CartpoleTestPlant::s_array new_state;
  for(int i = 0; i < 4; i++) {
    new_state(i) = i;
  }
  plant.setState(new_state);
  state = plant.getState();
  for(int i = 0; i < 4; i++) {
    EXPECT_EQ(state(i), i);
  }
}

TEST(BasePlant, getSetOptimizationStride) {
  CartpoleTestPlant plant;
  int optimization_stride = plant.getOptimizationStride();

  EXPECT_EQ(optimization_stride, 1);

  plant.setOptimizationStride(5);
  optimization_stride = plant.getOptimizationStride();

  EXPECT_EQ(optimization_stride, 5);
}

TEST(BasePlant, getSetDynamicsParams) {
  CartpoleTestPlant plant;

  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  CartpoleTestPlant::DYN_PARAMS_T params;

  params.cart_mass = 50;
  params.pole_length = 100;
  params.pole_mass = 150;

  plant.setDynamicsParams(params);
  EXPECT_EQ(plant.hasNewDynamicsParams(), true);

  CartpoleTestPlant::DYN_PARAMS_T new_params = plant.getNewDynamicsParams();
  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  EXPECT_EQ(new_params.cart_mass, params.cart_mass);
  EXPECT_EQ(new_params.pole_mass, params.pole_mass);
  EXPECT_EQ(new_params.pole_length, params.pole_length);
}


TEST(BasePlant, getSetCostParams) {
  CartpoleTestPlant plant;
  EXPECT_EQ(plant.hasNewCostParams(), false);

  CartpoleTestPlant::COST_PARAMS_T params;
  params.cart_position_coeff = 100;

  plant.setCostParams(params);
  EXPECT_EQ(plant.hasNewCostParams(), true);

  auto new_params = plant.getNewCostParams();
  EXPECT_EQ(plant.hasNewCostParams(), false);
  EXPECT_EQ(params.cart_position_coeff, new_params.cart_position_coeff);
}

typedef TestPlant<MockController> MockTestPlant;

TEST(BasePlant, runControlIterationStoppedTest) {
  // TODO look into mocking the dynamics and controller class
  MockController mockController;
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController.cost_ = &mockCost;
  mockController.model_ = &mockDynamics;

  MockTestPlant testPlant;

  EXPECT_CALL(mockController, computeControl(testing::_)).Times(0);

  std::atomic<bool> is_alive(false);
  testPlant.runControlIteration(&mockController, &is_alive);
}


TEST(BasePlant, runControlIterationDebugFalseTest) {
  // TODO look into mocking the dynamics and controller class
  MockController mockController;
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController.cost_ = &mockCost;
  mockController.model_ = &mockDynamics;

  MockTestPlant testPlant;

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(mockController, getFeedbackGains()).Times(0);
  EXPECT_CALL(mockController, getControlSeq()).Times(1);
  EXPECT_CALL(mockController, getStateSeq()).Times(1);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(&mockController, &is_alive);

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
  MockController mockController;
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController.cost_ = &mockCost;
  mockController.model_ = &mockDynamics;

  MockTestPlant testPlant;
  std::vector<int> map_desc;
  std::vector<float4> values;

  testPlant.setCostParams(mockCostParams());
  testPlant.setDynamicsParams(mockDynamicsParams());

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(1);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(1);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(mockController, getControlSeq()).Times(1);
  EXPECT_CALL(mockController, getStateSeq()).Times(1);
  EXPECT_CALL(mockController, getFeedbackGains()).Times(0);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(&mockController, &is_alive);

  EXPECT_EQ(testPlant.checkStatus(), 1);
  // TODO check solution
}

TEST(BasePlant, runControlIterationDebugNoDisplayTest) {
  // TODO look into mocking the dynamics and controller class
  MockController mockController;
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController.cost_ = &mockCost;
  mockController.model_ = &mockDynamics;

  MockTestPlant testPlant;
  testPlant.setDebugMode(true);

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(1);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(mockController, getControlSeq()).Times(1);
  EXPECT_CALL(mockController, getStateSeq()).Times(1);
  EXPECT_CALL(mockController, getFeedbackGains()).Times(0);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(&mockController, &is_alive);
}


TEST(BasePlant, runControlIterationDebugDisplayTest) {
  // TODO look into mocking the dynamics and controller class
  MockController mockController;
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController.cost_ = &mockCost;
  mockController.model_ = &mockDynamics;

  MockTestPlant testPlant;
  testPlant.setDebugMode(true);

  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(1).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(mockCost, getDebugDisplay(testing::_)).Times(1);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(mockController, slideControlSequence(1)).Times(1);
  EXPECT_CALL(mockController, computeControl(testing::_)).Times(1);
  EXPECT_CALL(mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(mockController, getControlSeq()).Times(1);
  EXPECT_CALL(mockController, getStateSeq()).Times(1);
  EXPECT_CALL(mockController, getFeedbackGains()).Times(0);

  std::atomic<bool> is_alive(true);
  testPlant.runControlIteration(&mockController, &is_alive);
}


TEST(BasePlant, runControlLoop) {
  MockController mockController;
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController.cost_ = &mockCost;
  mockController.model_ = &mockDynamics;

  MockTestPlant testPlant;
  int hz = testPlant.getHz();
  double time = 1; // in seconds

  // setup mock expected calls
  EXPECT_CALL(mockCost, getDebugDisplayEnabled()).Times(0);
  EXPECT_CALL(mockCost, setParams(testing::_)).Times(0);
  EXPECT_CALL(mockDynamics, setParams(testing::_)).Times(0);
  // TODO //EXPECT_CALL(mockCost, updateCostmap(map_desc, values)).Times(1);
  EXPECT_CALL(mockController, resetControls()).Times(1);
  EXPECT_CALL(mockController, slideControlSequence(1)).Times(hz/time);
  EXPECT_CALL(mockController, computeControl(testing::_)).Times(hz/time);
  EXPECT_CALL(mockController, computeFeedbackGains(testing::_)).Times(0);
  EXPECT_CALL(mockController, getFeedbackGains()).Times(0);
  EXPECT_CALL(mockController, getControlSeq()).Times(hz/time);
  EXPECT_CALL(mockController, getStateSeq()).Times(hz/time);

  std::atomic<bool> is_alive(true);
  boost::thread optimizer;
  optimizer = boost::thread(boost::bind(&MockTestPlant::runControlLoop, &testPlant, &mockController, &is_alive));

  std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> loop_duration = std::chrono::steady_clock::now() - loop_start;
  while(loop_duration.count() < time*1e3) {
    usleep(50);
    loop_duration = std::chrono::steady_clock::now() - loop_start;
  }
  is_alive.store(false);
  optimizer.join();


  EXPECT_EQ(testPlant.checkStatus(), 1);

  // create thread
  // run thread for a period of time
  // check that is ran X number of times and results
}
