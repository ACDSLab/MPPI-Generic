#include <gtest/gtest.h>
#include <ddp/ddp_model_wrapper.h>
#include <ddp/ddp_tracking_costs.h>
#include <ddp/ddp.h>
#include <eigen3/Eigen/Dense>
#include <instantiations/cartpole_mppi/cartpole_mppi.cuh>

class ModelWrapper_Test : public testing::Test {
public:
  CartpoleDynamics model = CartpoleDynamics(1, 1, 1);
  std::shared_ptr<ModelWrapperDDP<CartpoleDynamics>> ddp_model = std::make_shared<ModelWrapperDDP<CartpoleDynamics>>(&model);

  CartpoleDynamics::state_array state;
  CartpoleDynamics::control_array control;
};


TEST_F(ModelWrapper_Test, StateDerivative_1) {
  CartpoleDynamics::state_array result;
  CartpoleDynamics::state_array known_result;

  state << 0, 0, 0, 0;
  control << 0;
  result = ddp_model->f(state, control);
  known_result << 0, 0, 0, 0;

  ASSERT_EQ(known_result, result);
}

TEST_F(ModelWrapper_Test, StateDerivative_2) {
  CartpoleDynamics::state_array result;
  CartpoleDynamics::state_array known_result;

  state << 1, 2, 3, 4;
  control << 5;
  result = ddp_model->f(state, control);
  model.computeStateDeriv(state, control, known_result);

  ASSERT_EQ(known_result, result);
}

TEST_F(ModelWrapper_Test, Jacobian_1) {
  Eigen::Matrix<float, CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM + CartpoleDynamics::CONTROL_DIM> result;
  Eigen::Matrix<float, CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM + CartpoleDynamics::CONTROL_DIM> known_result;

  CartpoleDynamics::dfdx A = CartpoleDynamics::dfdx::Zero();
  CartpoleDynamics::dfdu B = CartpoleDynamics::dfdu::Zero();

  state << 1, 2, 3, 4;
  control << 5;

  model.computeGrad(state, control, A, B);

  known_result.leftCols<CartpoleDynamics::STATE_DIM>() = A;
  known_result.rightCols<CartpoleDynamics::CONTROL_DIM>() = B;
  result = ddp_model->df(state, control);

  ASSERT_EQ(known_result, result);
}

class TrackingCosts_Test : public testing::Test {
public:
  Eigen::Matrix<float, CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM> Q;
  Eigen::Matrix<float, CartpoleDynamics::CONTROL_DIM, CartpoleDynamics::CONTROL_DIM> R;
  Eigen::Matrix<float, CartpoleDynamics::STATE_DIM+CartpoleDynamics::CONTROL_DIM,
          CartpoleDynamics::STATE_DIM+CartpoleDynamics::CONTROL_DIM> QR;
  int num_timesteps;

  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<CartpoleDynamics>>> tracking_cost;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<CartpoleDynamics>>> terminal_cost;

  CartpoleDynamics::state_array state;
  CartpoleDynamics::control_array control;

protected:
  void SetUp() override {
    Q = Eigen::Matrix<float, CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM>::Identity();
    R = Eigen::Matrix<float, CartpoleDynamics::CONTROL_DIM, CartpoleDynamics::CONTROL_DIM>::Identity();
    QR = Eigen::Matrix<float, CartpoleDynamics::STATE_DIM+CartpoleDynamics::CONTROL_DIM,
            CartpoleDynamics::STATE_DIM+CartpoleDynamics::CONTROL_DIM>::Identity();
    num_timesteps = 100;

    tracking_cost = std::make_shared<TrackingCostDDP<ModelWrapperDDP<CartpoleDynamics>>>(Q,R,num_timesteps);
    terminal_cost = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<CartpoleDynamics>>>(Q);

    state.resize(CartpoleDynamics::STATE_DIM, 1);
    control.resize(CartpoleDynamics::CONTROL_DIM, 1);
  }
};

TEST_F(TrackingCosts_Test, ComputeCost) {
  state << 1, 2, 3, 4;
  control << 5;
  int timestep = 0;
  ASSERT_FLOAT_EQ(1+4+9+16+25, tracking_cost->c(state, control, timestep));
  ASSERT_FLOAT_EQ(1+4+9+16, terminal_cost->c(state));
}

TEST_F(TrackingCosts_Test, ComputeCostGradient) {
  Eigen::Matrix<float, 5,1> known_result;
  state << 1, 2, 3, 4;
  control << 5;
  known_result.block(0,0,4,1) = state;
  known_result.block(4,0,1,1) = control;
  std::cout << known_result << std::endl;
  int timestep = 0;
  ASSERT_EQ(QR*known_result, tracking_cost->dc(state, control, timestep)) << "Known:\n" << QR*known_result
  << "\nTracking cost: \n" << tracking_cost->dc(state, control, timestep);
  ASSERT_EQ(Q*state, terminal_cost->dc(state)) << "Known:\n" << Q*state
                                               << "\nTerminal cost: \n" << terminal_cost->dc(state);
}

TEST_F(TrackingCosts_Test, ComputeCostHessian) {
    state << 1, 2, 3, 4;
    control << 5;
    int timestep = 0;
    ASSERT_EQ(QR, tracking_cost->d2c(state, control, timestep));
    ASSERT_EQ(Q, terminal_cost->d2c(state));
}

TEST(DDPSolver_Test, CartpoleSwingUp) {

    // SETUP THE MPPI CONTROLLER
    CartpoleDynamics model = CartpoleDynamics(1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

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
    const int num_timesteps = 100;

    CartpoleDynamics::control_array control_var = CartpoleDynamics::control_array::Constant(5.0);

    auto controller = VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, num_timesteps, 2048, 64, 8>(&model, &cost,
                                                                                                       dt, max_iter, gamma, num_timesteps, control_var);
    CartpoleDynamics::state_array current_state = CartpoleDynamics::state_array::Zero();
    // Compute the control
    controller.computeControl(current_state);
//    nominal_state = controller.get
    // END MPPI CONTROLLER


//    util::DefaultLogger logger;
//    bool verbose = false;
//    int num_iterations = 5;
//    Eigen::MatrixXf Q;
//    Eigen::MatrixXf R;
//    Eigen::MatrixXf QR;
//    Q = Eigen::MatrixXf::Identity(CartpoleDynamics::STATE_DIM,CartpoleDynamics::STATE_DIM);
//    R = Eigen::MatrixXf::Identity(CartpoleDynamics::CONTROL_DIM,CartpoleDynamics::CONTROL_DIM);
//    QR = Eigen::MatrixXf(CartpoleDynamics::STATE_DIM+ CartpoleDynamics::CONTROL_DIM,
//                         CartpoleDynamics::STATE_DIM + CartpoleDynamics::CONTROL_DIM);
//    QR.template topLeftCorner<CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM>() = Q;
//    QR.template bottomRightCorner<CartpoleDynamics::CONTROL_DIM, CartpoleDynamics::CONTROL_DIM>() = R;
//
//
//    std::shared_ptr<ModelWrapperDDP<CartpoleDynamics>> ddp_model = std::make_shared<ModelWrapperDDP<CartpoleDynamics>>(&model);
//
//    std::shared_ptr<DDP<ModelWrapperDDP<CartpoleDynamics>>> ddp_solver_ =
//            std::make_shared<DDP<ModelWrapperDDP<CartpoleDynamics>>>(dt, num_timesteps, num_iterations, &logger, verbose);
//
//    std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<CartpoleDynamics>>> tracking_cost =
//            std::make_shared<TrackingCostDDP<ModelWrapperDDP<CartpoleDynamics>>>(Q,R,num_timesteps);
//
//    std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<CartpoleDynamics>>> terminal_cost =
//            std::make_shared<TrackingTerminalCost<ModelWrapperDDP<CartpoleDynamics>>>(QR);
//
//    tracking_cost->setTargets(nominal_state, nominal_control, num_timesteps);
//
//    Eigen::Matrix<float, CartpoleDynamics::STATE_DIM, 1> s;
//    s << 0.0, 0.0, 0.0, 0.0;
//
//    Eigen::MatrixXf control_traj = Eigen::MatrixXf::Zero(CartpoleDynamics::CONTROL_DIM, num_timesteps);
//
//    OptimizerResult<ModelWrapperDDP<CartpoleDynamics>> result_ = ddp_solver_->run(s, control_traj,
//                                                                       *ddp_model, *tracking_cost, *terminal_cost);
}

