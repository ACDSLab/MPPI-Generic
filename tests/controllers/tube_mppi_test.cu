#include <gtest/gtest.h>
#include <instantiations/cartpole_mppi/cartpole_mppi.cuh>

TEST(TubeMPPITest, Construction) {
    Cartpole model = Cartpole(0.1, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;
    float dt = 0.01;
    int max_iter = 10;
    float gamma = 0.5;

    std::array<float, Cartpole::CONTROL_DIM> control_var = {2.5};

    auto controller = TubeMPPIController<Cartpole, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost, dt, max_iter,
            gamma, 100, control_var);

    auto ddp_model = new ModelWrapperDDP<Cartpole>(&model);

    FAIL();
}