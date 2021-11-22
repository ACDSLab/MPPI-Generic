#include <gtest/gtest.h>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>

typedef QuadraticCost<DoubleIntegratorDynamics> DIQuadCost;

TEST(DIQuadraticCost, Constructor) {
  DIQuadCost();
}


class DITargetQuadraticCost : public ::Testing::Test {
  public:
    DIQuadCost cost;
    DIQuadCost::TEMPLATED_PARAMS cost_params;
    void SetUp() override {
      cost_params = cost.getParams();

    }
}

TEST(DIQuadraticCost, SimpleStateCostCPU) {
  DIQuadCost cost;
  auto cost_params = cost.getParams();
  cost_params.s_goal[0] = 1;
  cost_params.s_goal[1] = -5;
  cost_params.s_goal[2] = 0;
  cost_params.s_goal[3] = 0.5;
  cost.setParams(cost_params);

  DoubleIntegratorDynamics::state_array s;
  s << 0, 0, 0, 0;
  float state_cost = cost.computeStateCost(s);
  ASSERT_FLOAT_EQ(state_cost, 26.25);
}

TEST(DIQuadraticCost, LateStateCostCPU) {
  DIQuadCost cost;
  auto cost_params = cost.getParams();
  cost_params.s_goal[0] = 1;
  cost_params.s_goal[1] = -5;
  cost_params.s_goal[2] = 0;
  cost_params.s_goal[3] = 0.5;
  cost.setParams(cost_params);

  DoubleIntegratorDynamics::state_array s;
  s << 0, 0, 0, 0;
  float state_cost = cost.computeStateCost(s, 10);
  ASSERT_FLOAT_EQ(state_cost, 26.25);
}
