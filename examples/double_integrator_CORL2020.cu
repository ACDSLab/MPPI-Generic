#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_robust_cost.cuh>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/controllers/Tube-MPPI/tube_mppi_controller.cuh>
#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <cnpy.h>
#include <random> // Used to generate random noise for control trajectories

bool tubeFailure(float *s) {
  float inner_path_radius2 = 1.675*1.675;
  float outer_path_radius2 = 2.325*2.325;
  float radial_position = s[0]*s[0] + s[1]*s[1];
  if ((radial_position < inner_path_radius2) || (radial_position > outer_path_radius2)) {
    return true;
  } else {
    return false;
  }
}

using Dyn = DoubleIntegratorDynamics;
using SCost = DoubleIntegratorCircleCost;
using RCost = DoubleIntegratorRobustCost;
const int num_timesteps = 50;  // Optimization time horizon

typedef Eigen::Matrix<float, Dyn::STATE_DIM, num_timesteps> state_trajectory;

void saveTraj(const Eigen::Ref<const state_trajectory>& traj, int t, std::vector<float>& vec) {
  for (int i = 0; i < num_timesteps; i++) {
    for (int j = 0; j < Dyn::STATE_DIM; j++) {
      vec[t * num_timesteps * Dyn::STATE_DIM +
          i * Dyn::STATE_DIM + j] = traj(j, i);
    }
  }
}

void saveState(const Eigen::Ref<const Dyn::state_array >& state, int t, std::vector<float>& vec) {
  for (int j = 0; j < Dyn::STATE_DIM; j++) {
    vec[t * Dyn::STATE_DIM + j] = state(j);
  }
}

int main() {
  // Run the double integrator example on all the controllers with the SAME noise 20 times.


  // Create a random number generator
  // Random number generator for system noise
  std::mt19937 gen;  // Standard mersenne_twister_engine which will be seeded
  std::normal_distribution<float> normal_distribution;
  std::random_device rd;
  gen.seed(rd()); // Seed the RNG with a random number
  normal_distribution = std::normal_distribution<float>(0, 1);

  // Problem setup
  const int total_time_horizon = 5000;
  float dt = 0.02; // Timestep of dynamics propagation
  int max_iter = 3; // Maximum running iterations of optimization
  float lambda = 2; // Learning rate parameter
  float alpha = 0.0;
  Dyn nom_model;
  Dyn large_model(100);
  SCost cost;
  RCost robust_cost;

  // Set the cost parameters
  auto params = robust_cost.getParams();
  params.velocity_desired = 2;
  params.crash_cost = 100;
  robust_cost.setParams(params);

  // DDP cost parameters
  Eigen::MatrixXf Q;
  Eigen::MatrixXf Qf;
  Eigen::MatrixXf R;

  Q = 500*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);
  Q(2,2) = 100;
  Q(3,3) = 100;
  R = 1*Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::CONTROL_DIM,DoubleIntegratorDynamics::CONTROL_DIM);

  Qf = Eigen::MatrixXf::Identity(DoubleIntegratorDynamics::STATE_DIM,DoubleIntegratorDynamics::STATE_DIM);

  // Set the initial state
  DoubleIntegratorDynamics::state_array x_v, x_vl, x_t, x_rsc, x_rrc;
  x_v << 2, 0, 0, 1;
  x_vl << 2, 0, 0, 1;
  x_t << 2, 0, 0, 1;
  x_rsc << 2, 0, 0, 1;
  x_rrc << 2, 0, 0, 1;



  DoubleIntegratorDynamics::state_array xdot;

  // control variance
  DoubleIntegratorDynamics::control_array control_var;
  control_var << 1, 1;


  // Save actual trajectories, nominal_trajectory, free energy
  std::vector<float> van_trajectory(Dyn::STATE_DIM*total_time_horizon, 0);
  std::vector<float> van_nominal_traj(Dyn::STATE_DIM*num_timesteps*total_time_horizon, 0);
  std::vector<float> van_free_energy(total_time_horizon, 0);


  std::vector<float> van_large_trajectory(Dyn::STATE_DIM*total_time_horizon, 0);
  std::vector<float> van_large_nominal_traj(Dyn::STATE_DIM*num_timesteps*total_time_horizon, 0);
  std::vector<float> van_large_free_energy(total_time_horizon, 0);


  std::vector<float> tube_trajectory(Dyn::STATE_DIM*total_time_horizon, 0);
  std::vector<float> tube_nominal_traj(Dyn::STATE_DIM*num_timesteps*total_time_horizon, 0);
  std::vector<float> tube_nominal_free_energy(total_time_horizon, 0);
  std::vector<float> tube_real_free_energy(total_time_horizon, 0);
  std::vector<float> tube_nominal_state_used(total_time_horizon, 0);



  std::vector<float> robust_sc_trajectory(Dyn::STATE_DIM*total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_traj(Dyn::STATE_DIM*num_timesteps*total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_free_energy(total_time_horizon, 0);
  std::vector<float> robust_sc_real_free_energy(total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_sc_tube_real_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_sc_tube_real_free_energy_growth_bound(total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_state_used(total_time_horizon, 0);


  std::vector<float> robust_rc_trajectory(Dyn::STATE_DIM*total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_traj(Dyn::STATE_DIM*num_timesteps*total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_free_energy(total_time_horizon, 0);
  std::vector<float> robust_rc_real_free_energy(total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_rc_tube_real_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_rc_tube_real_free_energy_growth_bound(total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_state_used(total_time_horizon, 0);


  // Initialize the controllers
  auto nom_vanilla_controller = VanillaMPPIController<Dyn, SCost, num_timesteps,
          1024, 64, 8>(&nom_model, &cost, dt, max_iter, lambda, alpha, control_var);
  nom_vanilla_controller.initDDP(Q, Qf, R);


  auto large_vanilla_controller = VanillaMPPIController<Dyn, SCost, num_timesteps,
          1024, 64, 8>(&large_model, &cost, dt, max_iter, lambda, alpha, control_var);
  large_vanilla_controller.initDDP(Q, Qf, R);


  // Initialize the tube MPPI controller
  auto tube_controller = TubeMPPIController<Dyn,
          SCost, num_timesteps,
          1024, 64, 8>(&large_model, &cost, dt, max_iter, lambda, alpha, Q, Qf, R,
                                 control_var);
  tube_controller.setNominalThreshold(20);

  // Robust controller standard
  float value_function_threshold = 10.0;
  auto robust_sc_controller = RobustMPPIController<Dyn, SCost, num_timesteps,
          1024, 64, 8, 1>(&large_model, &cost, dt, max_iter, lambda, alpha, value_function_threshold, Q, Qf, R, control_var);

  // Robust controller robust
  value_function_threshold = 10.0;
  auto robust_rc_controller = RobustMPPIController<Dyn, RCost, num_timesteps,
          1024, 64, 8, 1>(&large_model, &robust_cost, dt, max_iter, lambda, alpha, value_function_threshold, Q, Qf, R, control_var);

  // Start the loop
  for (int t = 0; t < total_time_horizon; ++t) {
    // Compute universal state disturbance
    Dyn::state_array noise;
    noise << 0.0, 0.0, normal_distribution(gen), normal_distribution(gen);
    /********************** Vanilla **********************/
    {
      // Compute the control
      nom_vanilla_controller.computeControl(x_v, 1);

      // Compute the feedback gains
      nom_vanilla_controller.computeFeedbackGains(x_v);

      auto nominal_trajectory = nom_vanilla_controller.getStateSeq();
      auto nominal_control = nom_vanilla_controller.getControlSeq();
      nom_vanilla_controller.computeFeedbackPropagatedStateSeq();
      auto feedback_state_trajectory = nom_vanilla_controller.getFeedbackPropagatedStateSeq();
      auto fe_stat = nom_vanilla_controller.getFreeEnergyStatistics();

      // Save everything
      saveTraj(nominal_trajectory, t, van_nominal_traj);
      saveState(x_v, t, van_trajectory);
      van_free_energy[t] = fe_stat.real_sys.freeEnergyMean;


      // Get the open loop control
      DoubleIntegratorDynamics::control_array current_control = nominal_control.col(
              0);

      // Apply the feedback given the current state
      current_control += nom_vanilla_controller.getFeedbackGains()[0] *
                         (x_v - nominal_trajectory.col(0));

      // Propagate the state forward
      nom_model.computeDynamics(x_v, current_control, xdot);
      nom_model.updateState(x_v, xdot, dt);

      // Add disturbance
      x_v += noise*sqrt(nom_model.getParams().system_noise)*dt;

      // Slide the control sequence
      nom_vanilla_controller.slideControlSequence(1);
    }

    /********************** Vanilla Large **********************/


    /********************** Tube **********************/

    /********************** Robust Standard **********************/

    /********************** Robust Robust **********************/
  }

  /************* Save CNPY *********************/
  cnpy::npy_save("vanilla_state_trajectory.npy",van_trajectory.data(),
                 {total_time_horizon, DoubleIntegratorDynamics::STATE_DIM},"w");
  cnpy::npy_save("vanilla_nominal_trajectory.npy",van_nominal_traj.data(),
                 {total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM},"w");
  cnpy::npy_save("vanilla_free_energy.npy",van_free_energy.data(),
                 {total_time_horizon},"w");


    return 0;
}