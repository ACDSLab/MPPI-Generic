#include <gtest/gtest.h>

#include <mppi/cost_functions/cost_generic_kernel_tests.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>

template <class COST_T>
class GeneralCostTest : public ::testing::Test
{
public:
  COST_T cost;

protected:
  void SetUp() override
  {
  }

  void TearDown() override
  {
  }
};

// using DIFFERENT_COSTS = ::testing::Types<CartpoleQuadraticCost>;
using DIFFERENT_COSTS = ::testing::Types<DoubleIntegratorCircleCost, CartpoleQuadraticCost>;

TYPED_TEST_SUITE(GeneralCostTest, DIFFERENT_COSTS);

TYPED_TEST(GeneralCostTest, CPUvsGPURolloutCost)
{
  checkGPURolloutCost<TypeParam>(this->cost, 0.01);
}
