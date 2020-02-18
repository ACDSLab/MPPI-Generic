#include <gtest/gtest.h>
#include <ddp/ddp_model_wrapper.h>
#include <ddp/ddp_tracking_costs.h>
#include <ddp/ddp.h>
#include <eigen3/Eigen/Dense>
#include <dynamics/cartpole/cartpole_dynamics.cuh>

class ModelWrapper_Test : public testing::Test {
public:
    CartpoleDynamics model = CartpoleDynamics(0.01, 1, 1, 1);
    std::shared_ptr<ModelWrapperDDP<CartpoleDynamics>> ddp_model = std::make_shared<ModelWrapperDDP<CartpoleDynamics>>(&model);

    Eigen::MatrixXf state;
    Eigen::MatrixXf control;
    Eigen::MatrixXf result;
    Eigen::MatrixXf known_result;

protected:
    void SetUp() override {
        state.resize(CartpoleDynamics::STATE_DIM, 1);
        control.resize(CartpoleDynamics::CONTROL_DIM, 1);
    }
};


TEST_F(ModelWrapper_Test, StateDerivative_1) {
    result.resize(CartpoleDynamics::STATE_DIM, 1);
    known_result.resize(CartpoleDynamics::STATE_DIM, 1);

    state << 0, 0, 0, 0;
    control << 0;
    result = ddp_model->f(state, control);
    known_result << 0, 0, 0, 0;

    ASSERT_EQ(known_result, result);
}

TEST_F(ModelWrapper_Test, StateDerivative_2) {
    result.resize(CartpoleDynamics::STATE_DIM, 1);
    known_result.resize(CartpoleDynamics::STATE_DIM, 1);

    state << 1, 2, 3, 4;
    control << 5;
    result = ddp_model->f(state, control);
    model.computeStateDeriv(state, control, known_result);

    ASSERT_EQ(known_result, result);
}

TEST_F(ModelWrapper_Test, Jacobian_1) {
    result.resize(CartpoleDynamics::STATE_DIM,
            CartpoleDynamics::STATE_DIM+CartpoleDynamics::CONTROL_DIM);
    known_result.resize(CartpoleDynamics::STATE_DIM,
            CartpoleDynamics::STATE_DIM+CartpoleDynamics::CONTROL_DIM);

    known_result = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM,
            CartpoleDynamics::STATE_DIM + CartpoleDynamics::CONTROL_DIM);

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM,
            CartpoleDynamics::STATE_DIM);
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM,
            CartpoleDynamics::CONTROL_DIM);

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
    Eigen::MatrixXf Q;
    Eigen::MatrixXf R;
    Eigen::MatrixXf QR;
    int num_timesteps;

    std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<CartpoleDynamics>>> tracking_cost;
    std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<CartpoleDynamics>>> terminal_cost;

    Eigen::MatrixXf state;
    Eigen::MatrixXf control;
    Eigen::MatrixXf result;
    Eigen::MatrixXf known_result;


protected:
    void SetUp() override {
        Q = Eigen::MatrixXf::Identity(CartpoleDynamics::STATE_DIM,CartpoleDynamics::STATE_DIM);
        R = Eigen::MatrixXf::Identity(CartpoleDynamics::CONTROL_DIM,CartpoleDynamics::CONTROL_DIM);
        QR = Eigen::MatrixXf(CartpoleDynamics::STATE_DIM+ CartpoleDynamics::CONTROL_DIM,
        CartpoleDynamics::STATE_DIM + CartpoleDynamics::CONTROL_DIM);
        QR.template topLeftCorner<CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM>() = Q;
        QR.template bottomRightCorner<CartpoleDynamics::CONTROL_DIM, CartpoleDynamics::CONTROL_DIM>() = R;
        num_timesteps = 100;

        tracking_cost = std::make_shared<TrackingCostDDP<ModelWrapperDDP<CartpoleDynamics>>>(Q,R,num_timesteps);
        terminal_cost = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<CartpoleDynamics>>>(QR);

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
    known_result.resize(5,1);
    state << 1, 2, 3, 4;
    control << 5;
    known_result.block(0,0,4,1) = state;
    known_result.block(4,0,1,1) = control;
    std::cout << known_result << std::endl;
    int timestep = 0;
    ASSERT_EQ(QR*known_result, tracking_cost->dc(state, control, timestep)) << "Known:\n" << QR*known_result
    << "\nTracking cost: \n" << tracking_cost->dc(state, control, timestep);
    ASSERT_EQ(Q*state, terminal_cost->dc(state));
}

TEST_F(TrackingCosts_Test, ComputeCostHessian) {
    state << 1, 2, 3, 4;
    control << 5;
    int timestep = 0;
    ASSERT_EQ(QR, tracking_cost->d2c(state, control, timestep));
    ASSERT_EQ(Q, terminal_cost->d2c(state));
}