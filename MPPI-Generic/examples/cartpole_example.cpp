#include <controllers/cartpole_mppi.cuh>

int main(int argc, char** argv) {
    Cartpole model(0.1, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

    float dt = 0.01;
    int max_iter = 10;
    float gamma = 0.5;

    std::array<float, Cartpole::CONTROL_DIM> control_var = {2.5};

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 150, 64, 8>(&model, &cost,
            dt, max_iter, gamma,
            control_var);

    return 0;
}