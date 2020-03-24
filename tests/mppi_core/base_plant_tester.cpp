//
// Created by jgibson37 on 2/24/20.
//


#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <utils/test_helper.h>
#include <random>
#include <algorithm>
#include <numeric>

#include <mppi_core/base_plant.hpp>
#include <instantiations/cartpole_mppi/cartpole_mppi.cuh>

typedef VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8> cartpole_mppi_controller;

class TestPlant : public basePlant<cartpole_mppi_controller> {
  double getLastPoseTime() override {
    return 0.0;
  }

  void setTimingInfo(double avg_duration_ms, double avg_tick_duration, double avg_sleep_time) override {

  }

  void setDebugImage(cv::Mat debug_img) override {

  }

  void setSolution(const s_traj& state_seq,
                           const c_traj& control_seq,
                           const K_mat& feedback_gains,
                           double timestamp,
                           double loop_speed) override {

  }

  int checkStatus() override {
    return 1;
  }
};

class MockTestPlant : TestPlant {
public:
  MOCK_METHOD0(checkStatus, int());
};


TEST(BasePlant, Constructor) {
  TestPlant plant;

  EXPECT_EQ(plant.hasNewCostParams(), false);
  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  EXPECT_EQ(plant.hasNewModel(), false);
  EXPECT_EQ(plant.hasNewCostmap(), false);
  EXPECT_EQ(plant.hasNewObstacles(), false);
}

TEST(BasePlant, getAndSetState) {
  TestPlant plant;
  TestPlant::s_array state = plant.getState();
  // check initial state is zerod
  for(int i = 0; i < 4; i++) {
    EXPECT_EQ(state(i), 0.0);
  }

  TestPlant::s_array new_state;
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
  TestPlant plant;
  int optimization_stride = plant.getOptimizationStride();

  EXPECT_EQ(optimization_stride, 0);

  plant.setOptimizationStride(5);
  optimization_stride = plant.getOptimizationStride();

  EXPECT_EQ(optimization_stride, 5);
}

TEST(BasePlant, getSetDynamicsParams) {
  TestPlant plant;

  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  TestPlant::DYN_PARAMS_T params;

  params.cart_mass = 50;
  params.pole_length = 100;
  params.pole_mass = 150;

  plant.setDynamicsParams(params);
  EXPECT_EQ(plant.hasNewDynamicsParams(), true);

  TestPlant::DYN_PARAMS_T new_params = plant.getNewDynamicsParams();
  EXPECT_EQ(plant.hasNewDynamicsParams(), false);
  EXPECT_EQ(new_params.cart_mass, params.cart_mass);
  EXPECT_EQ(new_params.pole_mass, params.pole_mass);
  EXPECT_EQ(new_params.pole_length, params.pole_length);
}


TEST(BasePlant, getSetCostParams) {
  TestPlant plant;
  EXPECT_EQ(plant.hasNewCostParams(), false);

  TestPlant::COST_PARAMS_T params;
  params.cart_position_coeff = 100;

  plant.setCostParams(params);
  EXPECT_EQ(plant.hasNewCostParams(), true);

  auto new_params = plant.getNewCostParams();
  EXPECT_EQ(plant.hasNewCostParams(), false);
  EXPECT_EQ(params.cart_position_coeff, new_params.cart_position_coeff);
}

TEST(BasePlant, runControlLoopTest) {
  MockTestPlant mockPlant;

  EXPECT_CALL(mockPlant, checkStatus()).Times(1).WillOnce(testing::Return(10));

  EXPECT_EQ(mockPlant.checkStatus(), 5);

  // TODO look into mocking the dynamics and controller class
}

TEST(BasePlant, runControlIterationTest) {
  // TODO look into mocking the dynamics and controller class
}




