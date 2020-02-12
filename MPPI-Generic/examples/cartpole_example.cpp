#include <controllers/cartpole_mppi.cuh>

int main(int argc, char** argv) {
    Cartpole model(0.1, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

//  typedef VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 150, 64, 8>::control_array control_array;

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 150, 64, 8>(&model, &cost);

    CartpoleController;
    return 0;
}