#include <gtest/gtest.h>
#include <controllers/cartpole_mppi.cuh>


class Cartpole_VanillaMPPI: public ::testing::Test {
public:
    Cartpole model = Cartpole(0.1, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;
    float dt = 0.01;
    int max_iter = 10;
    float gamma = 0.5;
};



TEST_F(Cartpole_VanillaMPPI, BindToStream) {
    const int num_timesteps = 100;
    const int num_rollouts = 256;

    std::array<float, Cartpole::CONTROL_DIM> control_var = {2.5};
    std::array<float, Cartpole::CONTROL_DIM*num_timesteps> init_control = {0};
    cudaStream_t stream;

    HANDLE_ERROR(cudaStreamCreate(&stream));

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, num_timesteps, num_rollouts, 64, 8>(&model, &cost,
            dt, max_iter, gamma, control_var, init_control, stream);

    EXPECT_EQ(CartpoleController.stream_, CartpoleController.model_->stream_)
                        << "Stream bind to dynamics failure";
    EXPECT_EQ(CartpoleController.stream_, CartpoleController.cost_->stream_)
                        << "Stream bind to cost failure";
    HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST_F(Cartpole_VanillaMPPI, UpdateNoiseVariance) {
    const int num_timesteps = 150;
    const int num_rollouts = 512;
    std::array<float, Cartpole::CONTROL_DIM> control_var = {1.5};
    std::array<float, Cartpole::CONTROL_DIM> new_control_var = {3.5};

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, num_timesteps, num_rollouts, 64, 8>(&model, &cost,
            dt, max_iter, gamma, control_var);

    CartpoleController.updateControlNoiseVariance(new_control_var);

    EXPECT_FLOAT_EQ(new_control_var[0], CartpoleController.getControlVariance()[0]);
}


