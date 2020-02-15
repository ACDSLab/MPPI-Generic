#include <instantiations/cartpole_mppi/cartpole_mppi.cuh>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    CartpoleDynamics model(0.01, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

    cartpoleQuadraticCostParams new_params;
    new_params.cart_position_coeff = 100;
    new_params.pole_angle_coeff = 200;
    new_params.cart_velocity_coeff = 10;
    new_params.pole_angular_velocity_coeff = 20;
    new_params.control_force_coeff = 1;
    new_params.terminal_cost_coeff = 0;
    new_params.desired_terminal_state[0] = -20;
    new_params.desired_terminal_state[1] = 0;
    new_params.desired_terminal_state[2] = M_PI;
    new_params.desired_terminal_state[3] = 0;

    cost.setParams(new_params);


    float dt = 0.01;
    int max_iter = 1;
    float gamma = 0.25;
    int num_timesteps = 100;

    std::array<float, CartpoleDynamics::CONTROL_DIM> control_var = {5.0};

    auto CartpoleController = VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost,
                                                                                                               dt, max_iter, gamma, num_timesteps, control_var);

    decltype(CartpoleController)::state_array current_state { 0, 0, 0, 0};


    int time_horizon = 1000;

    float xdot[CartpoleDynamics::STATE_DIM];

    auto time_start = std::chrono::system_clock::now();
    for (int i =0; i < time_horizon; ++i) {
        if (i % 50 == 0) {
            printf("Current Time: %f    ", i * dt);
            printf("Current Baseline Cost: %f    ", CartpoleController.getBaselineCost());
            model.printState(current_state.data());
        }

        // Compute the control
        CartpoleController.computeControl(current_state);

        // Increment the state
        model.xDot(current_state.data(), &CartpoleController.getControlSeq()[0], xdot);
        model.incrementState(current_state.data(), xdot, dt);

        // Slide the controls down before calling the optimizer again
        CartpoleController.slideControlSequence(1);
    }
    auto time_end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration<double, std::milli>(time_end - time_start);
    printf("The elapsed time is: %f milliseconds", diff.count());
//    std::cout << "The current control at timestep " << i << " is: " << CartpoleController.get_control_seq()[i] << std::endl;

    return 0;
}
