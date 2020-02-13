#include <controllers/cartpole_mppi.cuh>
#include <iostream>

int main(int argc, char** argv) {
    Cartpole model(0.1, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

    float dt = 0.01;
    int max_iter = 10;
    float gamma = 0.5;

    std::array<float, Cartpole::CONTROL_DIM> control_var = {0.1};

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 150, 64, 8>(&model, &cost,
            dt, max_iter, gamma, control_var);

    decltype(CartpoleController)::state_array x0 { 0, 0, 0, 0};

    CartpoleController.computeControl(x0);

    for (int i =0; i < 100; ++i) {
        std::cout << "The current control at timestep " << i << " is: " << CartpoleController.get_control_seq()[i] << std::endl;
    }

    return 0;
}