#include <gtest/gtest.h>
#include <mppi/instantiations/cartpole_mppi/cartpole_mppi.cuh>
#include <mppi/instantiations/quadrotor_mppi/quadrotor_mppi.cuh>


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

  CartpoleDynamics::control_array control_std_dev = CartpoleDynamics::control_array::Constant(2.5);
  Eigen::Matrix<float, CartpoleDynamics::CONTROL_DIM, num_timesteps> init_control = Eigen::Matrix<float, CartpoleDynamics::CONTROL_DIM, num_timesteps>::Constant(0);
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  auto CartpoleController = new VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, num_timesteps, num_rollouts, 64, 8>
          (&model, &cost, dt, max_iter, gamma, control_std_dev, num_timesteps, init_control, stream);

  EXPECT_EQ(CartpoleController->stream_, CartpoleController->model_->stream_)
                      << "Stream bind to dynamics failure";
  EXPECT_EQ(CartpoleController->stream_, CartpoleController->cost_->stream_)
                      << "Stream bind to cost failure";
  HANDLE_ERROR(cudaStreamDestroy(stream));

  delete(CartpoleController);
}

TEST_F(Cartpole_VanillaMPPI, UpdateNoiseStdDev) {
  const int num_timesteps = 150;
  const int num_rollouts = 512;
  CartpoleDynamics::control_array control_std_dev = CartpoleDynamics::control_array::Constant(1.5);
  CartpoleDynamics::control_array new_control_std_dev = CartpoleDynamics::control_array::Constant(3.5);

  auto CartpoleController = new VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, num_timesteps, num_rollouts, 64, 8>(&model, &cost,
          dt, max_iter, gamma, control_std_dev);
  std::cout << sizeof(*CartpoleController) << std::endl;

  std::cout << CartpoleController->getControlStdDev() << std::endl;

  CartpoleController->updateControlNoiseStdDev(new_control_std_dev);

  std::cout << CartpoleController->getControlStdDev() << std::endl;

  EXPECT_FLOAT_EQ(new_control_std_dev[0], CartpoleController->getControlStdDev()[0]);
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

  CartpoleDynamics::control_array control_std_dev = CartpoleDynamics::control_array::Constant(5.0);

  auto controller = VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost,
                                                                                                     dt, max_iter, gamma, control_std_dev);
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
  CartpoleDynamics::control_array control_std_dev = CartpoleDynamics::control_array::Constant(5.0);
  auto controller = new VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>(&model, &cost,
                                                                                                     dt, max_iter, gamma, control_std_dev);

  CartpoleDynamics::state_array current_state = CartpoleDynamics::state_array::Zero();

  // Compute the control
  controller->computeControl(current_state);

  delete(controller);

}

class Quadrotor_VanillaMPPI: public ::testing::Test {
public:
  using DYN = QuadrotorDynamics;
  using COST = QuadrotorQuadraticCost;

  const int num_timesteps = 150;
  typedef VanillaMPPIController<DYN, COST, 150, 2048, 64, 8> CONTROLLER;
  DYN model;
  QuadrotorQuadraticCost cost;
};

TEST_F(Quadrotor_VanillaMPPI, HoverTest) {
  QuadrotorQuadraticCostParams new_params;
  new_params.x_goal()[2] = 1;
  new_params.x_coeff = 150;
  // new_params.q_coeff = 15;
  new_params.roll_coeff = 15;
  new_params.pitch_coeff = 15;
  new_params.yaw_coeff = 15;
  new_params.w_coeff = 5;
  float acceptable_distance = 0.15; // meters

  std::cout << "Goal: " << new_params.getDesiredState().transpose() << std::endl;

  cost.setParams(new_params);

  float dt = 0.01;
  int max_iter = 1;
  float gamma = 0.25;

  DYN::control_array control_std_dev = DYN::control_array::Constant(0.5);
  control_std_dev[3] = 5;

  CONTROLLER::control_trajectory init_control = CONTROLLER::control_trajectory::Zero();
  for(int i = 0; i < num_timesteps; i++) {
    init_control(4, i) = mppi_math::GRAVITY;
  }

  auto controller = CONTROLLER(&model,
                               &cost,
                               dt,
                               max_iter,
                               gamma,
                               control_std_dev,
                               num_timesteps,
                               init_control);
  controller.setDebug(true);
  DYN::state_array current_state = DYN::state_array::Zero();
  current_state(6) = 1;  // set q_w to 1
  int time_horizon = 3000;

  //float xdot[DYN::STATE_DIM];
  DYN::state_array xdot(4, 1);
  DYN::control_array control;
  control = controller.getControlSeq().block(0, 0, DYN::CONTROL_DIM, 1);

  int far_away_cnt = 0;
  Eigen::Vector3f goal_pos = new_params.getDesiredState().block<3, 1>(0, 0);

  for (int i =0; i < time_horizon; ++i) {
    if (i % 50 == 0) {
      printf("Current Time: %6.2f    ", i * dt);
      printf("Current Baseline Cost: %f \n", controller.getBaselineCost());
      printf("State Cost: %f\n", controller.cost_->computeStateCost(current_state));
      model.printState(current_state.data());
      std::cout << "Control: " << control.transpose() << std::endl;
    }
    if (std::isnan(controller.getBaselineCost()) || control.hasNaN() ||
        current_state.hasNaN()) {
      printf("ENCOUNTERED A NAN!!\n");
      printf("Current Time: %f    ", i * dt);
      printf("Current Baseline Cost: %f \n", controller.getBaselineCost());
      model.printState(current_state.data());
      std::cout << "Control: " << control.transpose() << std::endl;

      break;
    }

    // Compute the control
    controller.computeControl(current_state);

    control = controller.getControlSeq().block(0, 0, DYN::CONTROL_DIM, 1);
    // Increment the state
    model.computeStateDeriv(current_state, control, xdot);
    model.updateState(current_state, xdot, dt);

    controller.slideControlSequence(1);
    Eigen::Vector3f pos = current_state.block<3, 1>(0, 0);
    float dist = (pos -goal_pos).norm();
    if (dist > acceptable_distance) {
      far_away_cnt++;
    }

  }
  float percentage_in_ball = float(far_away_cnt) / time_horizon;
  std::cout << "Amount of time outside " << acceptable_distance << " m: "
            << percentage_in_ball * 100 << "%"<< std::endl;

  EXPECT_LT(percentage_in_ball, 0.1);
}