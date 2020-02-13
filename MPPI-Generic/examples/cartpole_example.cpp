#include <controllers/cartpole_mppi.cuh>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    Cartpole model(0.01, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

    cartpoleQuadraticCostParams new_params;
    new_params.desired_terminal_state[0] = 2;
    new_params.desired_terminal_state[1] = 0;
    new_params.desired_terminal_state[2] = M_PI;
    new_params.desired_terminal_state[3] = 0;

    cost.setParams(new_params);


    float dt = 0.01;
    int max_iter = 1;
    float gamma = 0.25;

    std::array<float, Cartpole::CONTROL_DIM> control_var = {5.0};

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost,
            dt, max_iter, gamma, control_var);

    decltype(CartpoleController)::state_array current_state { 0, 0, 0, 0};


    int time_horizon = 500;

    float xdot[Cartpole::STATE_DIM];

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
        model.xDot(current_state.data(), &CartpoleController.get_control_seq()[0], xdot);
        model.incrementState(current_state.data(), xdot, dt);
    }
    auto time_end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration<double, std::milli>(time_end - time_start);
    printf("The elapsed time is: %f milliseconds", diff.count());
//    std::cout << "The current control at timestep " << i << " is: " << CartpoleController.get_control_seq()[i] << std::endl;

    return 0;
}