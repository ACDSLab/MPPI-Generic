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

    known_result = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM + CartpoleDynamics::CONTROL_DIM);

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM);
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM, CartpoleDynamics::CONTROL_DIM);

//    A = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM, CartpoleDynamics::STATE_DIM);
//    B = Eigen::MatrixXf::Zero(CartpoleDynamics::STATE_DIM, CartpoleDynamics::CONTROL_DIM);

    state << 1, 2, 3, 4;
    control << 5;

//    std::cout << "A prev matrix:" << std::endl;
//    std::cout << A << std::endl;
//    std::cout << "B prev matrix:" << std::endl;
//    std::cout << B << std::endl;

    model.computeGrad(state, control, A, B);
//    std::cout << "A matrix:" << std::endl;
//    std::cout << A << std::endl;
//    std::cout << "B matrix:" << std::endl;
//    std::cout << B << std::endl;


//    std::cout << "Known prev Result:" << std::endl;
//    std::cout << known_result << std::endl;

    known_result.leftCols<CartpoleDynamics::STATE_DIM>() = A;
    known_result.rightCols<CartpoleDynamics::CONTROL_DIM>() = B;
    result = ddp_model->df(state, control);

//    std::cout << "Known Result:" << std::endl;
//    std::cout << known_result << std::endl;
//    std::cout << "Result:" << std::endl;
//    std::cout << result << std::endl;

    ASSERT_EQ(known_result, result);
}