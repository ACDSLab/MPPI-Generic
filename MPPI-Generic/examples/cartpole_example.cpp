#include <controllers/cartpole_mppi.cuh>

int main(int argc, char** argv) {
    Cartpole model(0.1, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

    float dt = 0.01;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(1);

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 150, 64, 8>(&model, &cost, dt, stream);

    CartpoleController;
    return 0;
}