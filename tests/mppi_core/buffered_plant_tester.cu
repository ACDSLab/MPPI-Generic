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

template <class CONTROLLER_T>
class TestPlant : public BufferedPlant<CONTROLLER_T> {
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
          : BufferedPlant<CONTROLLER_T>(controller,hz,opt_stride) {
    this->buffer_time_horizon_ = 2.0;
    this->buffer_tau_ = 0.5;
    this->buffer_dt_ = 0.2;
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
  std::list<std::pair<typename BufferedPlant<CONTROLLER_T>::StateArray, double>> getBuffer() {return this->prev_states_;}
};

typedef TestPlant<MockController> MockTestPlant;

TEST(BufferedPlant, EmptyCheck) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);

  EXPECT_EQ(plant.getLatestTimeInBuffer(), -1);
  EXPECT_EQ(plant.getEarliestTimeInBuffer(), -1);
}

TEST(BufferedPlant, UpdateStateCheck) {
  std::shared_ptr<MockController> mockController = std::make_shared<MockController>();
  MockCost mockCost;
  MockDynamics mockDynamics;
  mockController->cost_ = &mockCost;
  mockController->model_ = &mockDynamics;

  MockTestPlant plant(mockController);

  MockDynamics::state_array state = MockDynamics::state_array::Zero();
  EXPECT_EQ(plant.getBufferSize(), 0);
  plant.updateState(state, 8.534);
  EXPECT_EQ(plant.getBufferSize(), 1);
  EXPECT_EQ(plant.getEarliestTimeInBuffer(), 8.534);
  EXPECT_EQ(plant.getLatestTimeInBuffer(), 8.534);

  plant.updateState(state, 9.09);
  EXPECT_EQ(plant.getBufferSize(), 2);
  EXPECT_EQ(plant.getEarliestTimeInBuffer(), 8.534);
  EXPECT_EQ(plant.getLatestTimeInBuffer(), 9.09);

  plant.updateState(state, 9.08);
  EXPECT_EQ(plant.getBufferSize(), 2);
  EXPECT_EQ(plant.getEarliestTimeInBuffer(), 8.534);
  EXPECT_EQ(plant.getLatestTimeInBuffer(), 9.09);

  plant.updateState(state, 10.535);
  EXPECT_EQ(plant.getBufferSize(), 2);
  EXPECT_EQ(plant.getEarliestTimeInBuffer(), 9.09);
  EXPECT_EQ(plant.getLatestTimeInBuffer(), 10.535);
}
