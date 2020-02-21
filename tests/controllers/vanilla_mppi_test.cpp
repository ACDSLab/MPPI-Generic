#include <gtest/gtest.h>
#include <instantiations/cartpole_mppi/cartpole_mppi.cuh>


class Cartpole_VanillaMPPI: public ::testing::Test {
public:
    CartpoleDynamics model = CartpoleDynamics(1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;
    float dt = 0.01;
    int max_iter = 10;
    float gamma = 0.5;
};



TEST_F(Cartpole_VanillaMPPI, BindToStream) {
    const int num_timesteps = 100;
    const int num_rollouts = 256;

    CartpoleDynamics::control_array control_var = CartpoleDynamics::control_array::Constant(2.5);
    Eigen::Matrix<float, CartpoleDynamics::CONTROL_DIM, num_timesteps> init_control = Eigen::Matrix<float, CartpoleDynamics::CONTROL_DIM, num_timesteps>::Constant(0);
    cudaStream_t stream;

    HANDLE_ERROR(cudaStreamCreate(&stream));

    auto CartpoleController = new VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, num_timesteps, num_rollouts, 64, 8>(&model, &cost,
                                                                                                                                 dt, max_iter, gamma, num_timesteps, control_var, init_control, stream);

    EXPECT_EQ(CartpoleController->stream_, CartpoleController->model_->stream_)
                        << "Stream bind to dynamics failure";
    EXPECT_EQ(CartpoleController->stream_, CartpoleController->cost_->stream_)
                        << "Stream bind to cost failure";
    HANDLE_ERROR(cudaStreamDestroy(stream));

    delete(CartpoleController);
}

TEST_F(Cartpole_VanillaMPPI, UpdateNoiseVariance) {
    const int num_timesteps = 150;
    const int num_rollouts = 512;
    CartpoleDynamics::control_array control_var = CartpoleDynamics::control_array::Constant(1.5);
    CartpoleDynamics::control_array new_control_var = CartpoleDynamics::control_array::Constant(3.5);

    auto CartpoleController = VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, num_timesteps, num_rollouts, 64, 8>(&model, &cost,
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

  CartpoleDynamics::control_array control_var = CartpoleDynamics::control_array::Constant(5.0);

  auto controller = VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost,
                                                                                                     dt, max_iter, gamma, num_timesteps, control_var);
  CartpoleDynamics::state_array current_state = CartpoleDynamics::state_array::Zero();
  int time_horizon = 1000;

  //float xdot[CartpoleDynamics::STATE_DIM];
  CartpoleDynamics::state_array xdot(4, 1);

  for (int i =0; i < time_horizon; ++i) {
    if (i % 50 == 0) {
      printf("Current Time: %f    ", i * dt);
      printf("Current Baseline Cost: %f    ", controller.getBaselineCost());
      model.printState(current_state.data());
    }

    // Compute the control
    controller.computeControl(current_state);

    CartpoleDynamics::control_array control;
    control = controller.getControlSeq().block(0, 0, CartpoleDynamics::CONTROL_DIM, 1);
    // Increment the state
    model.computeStateDeriv(current_state, control, xdot);
    model.updateState(current_state, xdot, dt);

    controller.slideControlSequence(1);

  }
  EXPECT_LT(controller.getBaselineCost(), 1.0);
}

TEST_F(Cartpole_VanillaMPPI, ConstructWithNew) {
  float dt = 0.01;
  int max_iter = 1;
  float gamma = 0.25;
  int num_timesteps = 100;
  CartpoleDynamics::control_array control_var = CartpoleDynamics::control_array::Constant(5.0);
  auto controller = new VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost,
                                                                                                     dt, max_iter, gamma, num_timesteps, control_var);

  CartpoleDynamics::state_array current_state = CartpoleDynamics::state_array::Zero();

  // Compute the control
  controller->computeControl(current_state);

  delete(controller);

}