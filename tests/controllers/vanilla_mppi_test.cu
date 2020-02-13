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
            dt, max_iter, gamma, num_timesteps, control_var, init_control, stream);

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
            dt, max_iter, gamma, num_timesteps, control_var);

    CartpoleController.updateControlNoiseVariance(new_control_var);

    EXPECT_FLOAT_EQ(new_control_var[0], CartpoleController.getControlVariance()[0]);
}

TEST_F(Cartpole_VanillaMPPI, SwingUpTest) {
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

    std::array<float, Cartpole::CONTROL_DIM> control_var = {5.0};

    auto controller = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost,
                                                                                                dt, max_iter, gamma, num_timesteps, control_var);
    decltype(controller)::state_array current_state {0, 0, 0, 0};
    int time_horizon = 1000;

    float xdot[Cartpole::STATE_DIM];

    auto time_start = std::chrono::system_clock::now();
    for (int i =0; i < time_horizon; ++i) {
        if (i % 50 == 0) {
            printf("Current Time: %f    ", i * dt);
            printf("Current Baseline Cost: %f    ", controller.getBaselineCost());
            model.printState(current_state.data());
        }

        // Compute the control
        controller.computeControl(current_state);

        // Increment the state
        model.xDot(current_state.data(), &controller.get_control_seq()[0], xdot);
        model.incrementState(current_state.data(), xdot, dt);
    }

    EXPECT_LT(controller.getBaselineCost(), 10.0);
}


