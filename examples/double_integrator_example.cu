#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

#include <iomanip>
// #include <sstream>

const int TIMESTEPS = 65;
const int NUM_ROLLOUTS = 128;

using DYN = DoubleIntegratorDynamics;
using COST = QuadraticCost<DYN>;
using FB_CONTROLLER = DDPFeedback<DYN, TIMESTEPS>;

int main()
{
  float dt = 0.015;

  // Set the initial state
  DYN::state_array x;
  x << -9, -9, 0.1, 0.1;
  DYN::state_array xdot;

  // control standard deviation
  DYN::control_array control_std_dev;
  control_std_dev << 0.1, 0.1;

  // Create the dynamics, cost function, and feedback controller
  DYN model;
  COST cost;
  FB_CONTROLLER fb_controller = FB_CONTROLLER(&model, dt);
  auto cost_params = cost.getParams();

  // Set up cost function
  DYN::state_array x_goal;
  x_goal << -4, -4, 0, 0;
  DYN::state_array q_coeffs;
  q_coeffs << 0.5, 0.5, 0.0, 0.0;
  for (int i = 0; i < DYN::STATE_DIM; i++)
  {
    cost_params.s_coeffs[i] = q_coeffs[i];
    cost_params.s_goal[i] = x_goal[i];
  }
  cost.setParams(cost_params);

  // Create MPPI Controller
  float lambda = 1;
  float alpha = 1.0;
  int max_iter = 1;
  int total_time_horizon = 300;

  auto controller = VanillaMPPIController<DYN, COST, FB_CONTROLLER, TIMESTEPS, NUM_ROLLOUTS, 64, 1>(
      &model, &cost, &fb_controller, dt, max_iter, lambda, alpha, control_std_dev);

  /********************** Vanilla MPPI **********************/
  float cumulative_cost = 0;
  int crash = 0;
  for (int t = 0; t < total_time_horizon; ++t)
  {
    // Compute the control
    controller.computeControl(x, 1);

    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    DYN::control_array current_control = nominal_control.col(0);
    // Propagate dynamics forward
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);
    if (t % 10 == 0)
    {
      std::cout << "T: " << std::fixed << std::setprecision(3) << t * dt;
      // << "s Free Energy: " << fe_stat.real_sys.freeEnergyMean
      // << " +- " << fe_stat.real_sys.freeEnergyVariance << std::endl;
      std::cout << " X: " << x.transpose() << std::endl;
    }

    // Slide the control sequence
    controller.slideControlSequence(1);
    cumulative_cost +=
        cost.computeRunningCost(x, current_control, current_control, control_std_dev, lambda, alpha, t, &crash);
  }
  std::cout << "Total Cost: " << cumulative_cost << std::endl;

  return 0;
}
