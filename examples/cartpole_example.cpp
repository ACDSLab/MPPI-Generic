#include <mppi/instantiations/cartpole_mppi/cartpole_mppi.cuh>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
  auto model = new CartpoleDynamics(1.0, 1.0, 1.0);
  auto cost = new CartpoleQuadraticCost;

  model->control_rngs_->x = -1;
  model->control_rngs_->y = 1;

  CartpoleQuadraticCostParams new_params;
  new_params.cart_position_coeff = 50;
  new_params.pole_angle_coeff = 200;
  new_params.cart_velocity_coeff = 10;
  new_params.pole_angular_velocity_coeff = 1;
  new_params.control_cost_coeff[0] = 0;
  new_params.terminal_cost_coeff = 0;
  new_params.desired_terminal_state[0] = 20;
  new_params.desired_terminal_state[1] = 0;
  new_params.desired_terminal_state[2] = M_PI;
  new_params.desired_terminal_state[3] = 0;

  cost->setParams(new_params);

  float dt = 0.02;
  int max_iter = 1;
  float lambda = 0.25;
  float alpha = 0.0;
  const int num_timesteps = 100;

  CartpoleDynamics::control_array control_var;
  control_var = CartpoleDynamics::control_array::Constant(1.0);

  // Feedback Controller
  auto fb_controller = new DDPFeedback<CartpoleDynamics, num_timesteps>(model, dt);

  auto CartpoleController = new VanillaMPPIController<CartpoleDynamics,
                                                      CartpoleQuadraticCost,
                                                      DDPFeedback<CartpoleDynamics, num_timesteps>,
                                                      num_timesteps,
                                                      2048,  64,  8>(model,
                                                                     cost,
                                                                     fb_controller,
                                                                     dt,
                                                                     max_iter,
                                                                     lambda,
                                                                     alpha,
                                                                     control_var);

  CartpoleDynamics::state_array current_state = CartpoleDynamics::state_array::Zero();

  int time_horizon = 5000;

  CartpoleDynamics::state_array xdot = CartpoleDynamics::state_array::Zero();

  auto time_start = std::chrono::system_clock::now();
  for (int i =0; i < time_horizon; ++i) {
    // Compute the control
    CartpoleController->computeControl(current_state, 1);

    // Increment the state
    CartpoleDynamics::control_array control;
    control = CartpoleController->getControlSeq().block(0, 0, CartpoleDynamics::CONTROL_DIM, 1);
    model->enforceConstraints(current_state, control);
    model->computeStateDeriv(current_state, control, xdot);
    model->updateState(current_state, xdot, dt);

    if (i % 50 == 0) {
      printf("Current Time: %f    ", i * dt);
      printf("Current Baseline Cost: %f    ", CartpoleController->getBaselineCost());
      model->printState(current_state.data());
//      std::cout << control << std::endl;
    }

    // Slide the controls down before calling the optimizer again
    CartpoleController->slideControlSequence(1);
  }
  auto time_end = std::chrono::system_clock::now();
  auto diff = std::chrono::duration<double, std::milli>(time_end - time_start);
  printf("The elapsed time is: %f milliseconds\n", diff.count());
//    std::cout << "The current control at timestep " << i << " is: " << CartpoleController.get_control_seq()[i] << std::endl;

  // cost->freeCudaMem();
  delete(CartpoleController);
  delete(cost);
  delete(model);

  return 0;
}
