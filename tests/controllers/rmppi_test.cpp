#include <gtest/gtest.h>
#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>

TEST(RMPPITest, CPURolloutKernel) {
  DoubleIntegratorDynamics model;
  DoubleIntegratorCircleCost cost;
  float dt = 0.01;
  int max_iter = 10;
  float gamma = 0.5;
  const int num_timesteps = 100;
}

/******************************************************************************
 * Test class for RobustControllerPrivateMethods
 ******************************************************************************/
class TestRobust: public RobustMPPIController<
         DoubleIntegratorDynamics, DoubleIntegratorCircleCost, 100, 512, 64, 8>{
 public:
  TestRobust(DoubleIntegratorDynamics *model, DoubleIntegratorCircleCost *cost) :
  RobustMPPIController(model, cost) {}

  int accessNumIters() {return num_iters_;};
 };

TEST(RMPPITest, InitEvalSelection_Weights) {
  /*
   * This test will ensure that the line search process to select the
   * number of points for free energy evaluation is correct.
   */
  using dynamics = DoubleIntegratorDynamics;
  using cost_function = DoubleIntegratorCircleCost;
  dynamics model(10);  // Initialize the double integrator dynamics
  cost_function cost;  // Initialize the cost function

  dynamics::state_array x_star_k, x_star_kp1, x_kp1;
  x_star_k << -4 , 0, 0, 0;
  x_star_kp1 << 4, 0, 0, 0;
  x_kp1 << 4, 4, 0, 0;

  const int num_candidates = 9;

  Eigen::Matrix<float, 2, num_candidates> known_candidates;
  known_candidates << -4 , -3, -2, -1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4, 4, 4;

  std::cout << known_candidates << std::endl;

  // Create the controller so we can test functions from it -> this should be a fixture
  auto controller = TestRobust(&model, &cost);

  std::cout << controller.accessNumIters() << std::endl;


}