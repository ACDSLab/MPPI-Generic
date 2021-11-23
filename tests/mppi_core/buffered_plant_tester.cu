//
// Created by jgibson37 on 2/24/20.
//


#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <numeric>

#include <mppi/core/buffered_plant.hpp>
#include <mppi/instantiations/cartpole_mppi/cartpole_mppi.cuh>
#include <mppi_test/mock_classes/mock_dynamics.h>
#include <mppi_test/mock_classes/mock_controller.h>
#include <mppi_test/mock_classes/mock_costs.h>

template <class CONTROLLER_T, int BUFFER_LENGTH>
class TestPlant : public BufferedPlant<CONTROLLER_T, BUFFER_LENGTH> {
public:
  double time_ = 0.0;

  double avgDurationMs_ = 0;
  double avgTickDuration_ = 0;
  double avgSleepTime_ = 0;

  using c_array = typename CONTROLLER_T::control_array;
  using c_traj = typename CONTROLLER_T::control_trajectory;

  using s_array = typename CONTROLLER_T::state_array;
  using s_traj = typename CONTROLLER_T::state_trajectory;

  using DYN_T = typename CONTROLLER_T::TEMPLATED_DYNAMICS;
  using DYN_PARAMS_T = typename DYN_T::DYN_PARAMS_T;
  using COST_T = typename CONTROLLER_T::TEMPLATED_COSTS;
  using COST_PARAMS_T = typename COST_T::COST_PARAMS_T;
  double timestamp_;
  double loop_speed_;

  TestPlant(std::shared_ptr<MockController> controller, double buffer_time_horizon=0.2, int hz = 20, int opt_stride=1)
          : BufferedPlant<CONTROLLER_T, BUFFER_LENGTH>(controller,hz,opt_stride) {
    this->buffer_time_horizon_ = buffer_time_horizon;
    this->buffer_tau_ = 0.2;
    this->buffer_dt_ = 0.02;
    controller->setDt(this->buffer_dt_);
  }


  void pubControl(const c_array& u) override {}

  void pubNominalState(const s_array& s) override {}

  void pubStateDivergence(const s_array& s) override {}

  void pubFreeEnergyStatistics(MPPIFreeEnergyStatistics& fe_stats) override {}

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
  double getBufferTimeHorizon() {return this->buffer_time_horizon_;}
  double getBufferTau() {return this->buffer_time_horizon_;}
  double getBufferDt() {return this->buffer_time_horizon_;}
  std::list<std::pair<typename BufferedPlant<CONTROLLER_T, BUFFER_LENGTH>::StateArray, double>> getBuffer() {
    return this->prev_states_;
  }
  void setLastTime(double time) {
    time_ = time;
    this->last_used_pose_update_time_ = time;
  }

};

typedef TestPlant<MockController, 10> MockTestPlant;

class BufferedPlantTest : public ::testing::Test {
protected:
  void SetUp() override {
    mockController = std::make_shared<MockController>();
    mockFeedback = new FEEDBACK_T(&mockDynamics, mockController->getDt());
    mockController->cost_ = &mockCost;
    mockController->model_ = &mockDynamics;
    mockController->fb_controller_ = mockFeedback;

    EXPECT_CALL(*mockController->cost_, getParams()).Times(1);
    EXPECT_CALL(*mockController->model_, getParams()).Times(1);

    plant = std::make_shared<MockTestPlant>(mockController);
  }

  void TearDown() override {
    plant = nullptr;
    mockController = nullptr;
    delete mockFeedback;
  }
  MockDynamics mockDynamics;
  MockCost mockCost;
  FEEDBACK_T* mockFeedback;
  std::shared_ptr<MockController> mockController;
  std::shared_ptr<MockTestPlant> plant;
};

TEST_F(BufferedPlantTest, EmptyCheck) {
  EXPECT_EQ(plant->getLatestTimeInBuffer(), -1);
  EXPECT_EQ(plant->getEarliestTimeInBuffer(), -1);
}

TEST_F(BufferedPlantTest, UpdateStateCheck) {
  EXPECT_CALL(*mockController, getCurrentControl(testing::_, testing::_, testing::_, testing::_, testing::_)).Times(5);

  plant->setLastTime(8.0);

  MockDynamics::state_array state = MockDynamics::state_array::Zero();
  EXPECT_EQ(plant->getBufferSize(), 0);
  plant->updateState(state, 8.0);
  EXPECT_EQ(plant->getBufferSize(), 1);
  EXPECT_EQ(plant->getEarliestTimeInBuffer(), 8.0);
  EXPECT_EQ(plant->getLatestTimeInBuffer(), 8.0);

  plant->updateState(state, 8.15);
  EXPECT_EQ(plant->getBufferSize(), 2);
  EXPECT_EQ(plant->getEarliestTimeInBuffer(), 8.0);
  EXPECT_EQ(plant->getLatestTimeInBuffer(), 8.15);

  plant->updateState(state, 8.25);
  EXPECT_EQ(plant->getBufferSize(), 2);
  EXPECT_EQ(plant->getEarliestTimeInBuffer(), 8.15);
  EXPECT_EQ(plant->getLatestTimeInBuffer(), 8.25);
}

TEST_F(BufferedPlantTest, getBufferTestZeros) {
  plant->setLastTime(10.0);

  EXPECT_CALL(*mockController, getCurrentControl(testing::_, testing::_, testing::_, testing::_, testing::_)).Times(41);

  MockDynamics::state_array state = MockDynamics::state_array::Zero();
  for(double i = 10;  i < 10.2; i+=0.01) {
    plant->updateState(state, i);
  }
  auto buffer = plant->getSmoothedBuffer();
  for(int row = 0; row < buffer.rows(); row++) {
    for(int col = 0; col < buffer.cols(); col++) {
      EXPECT_DOUBLE_EQ(buffer(row,col), 0) << "row " << row << " col " << col;
    }
  }
}

TEST_F(BufferedPlantTest, getBufferTestValues) {
  plant->setLastTime(10.0);
  EXPECT_CALL(*mockController, getCurrentControl(testing::_, testing::_, testing::_, testing::_, testing::_)).Times(39);

  std::array<double, 20> old_times = {10.0, 10.010526315789473, 10.021052631578947, 10.031578947368422, 10.042105263157895, 10.052631578947368, 10.063157894736841, 10.073684210526315, 10.08421052631579, 10.094736842105263, 10.105263157894736, 10.11578947368421, 10.126315789473685, 10.136842105263158, 10.147368421052631, 10.157894736842104, 10.168421052631578, 10.178947368421053, 10.189473684210526, 10.2};
  std::array<float, 20> y = {0.9167042154371116, 0.3776076510955664, 0.08226566905023869, 0.9551211742263026, 0.7253182130148879, 0.4865343940849741, 0.818409147529944, 0.24277620212257367, 0.8347730401736206, 0.6951747693420071, 0.20670429250120048, 0.20936316003591626, 0.3272321712567512, 0.20917661559581946, 0.25748266945151754, 0.11603616519900317, 0.5984983071353864, 0.22721356931144365, 0.46368822631629447, 0.020616505178830402};
  std::array<double, 11> T_new = {10.0, 10.02, 10.04, 10.06, 10.08, 10.1, 10.12, 10.139999999999999, 10.16, 10.18, 10.2};
  std::array<float, 11> result = {0.879620914876778, 0.3461727105285413, 0.6632283112147902, 0.7316952972942005, 0.5677906449134396, 0.4130980048084355, 0.29535833028409375, 0.2144932071325534, 0.27298944852590057, 0.4386378568779821, 0.03787996418981846};

  for(int i = 0; i < 20; i++) {
    MockDynamics::state_array state = MockDynamics::state_array::Zero();
    state(0, 0) = y[i];
    plant->updateState(state, old_times[i]);
  }
  EXPECT_EQ(plant->getBufferSize(), 20);
  auto buffer = plant->getSmoothedBuffer();

  for(int col = 0; col < buffer.cols(); col++) {
    EXPECT_FLOAT_EQ(buffer(0,col), result[col]);
  }
}
