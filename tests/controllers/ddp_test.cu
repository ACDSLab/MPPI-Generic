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
        result.resize(CartpoleDynamics::STATE_DIM, 1);
        known_result.resize(CartpoleDynamics::STATE_DIM, 1);
    }
};


TEST_F(ModelWrapper_Test, StateDerivative_1) {

    state << 0, 0, 0, 0;
    control << 0;
    result = ddp_model->f(state, control);
    known_result << 0, 0, 0, 0;

    ASSERT_EQ(known_result, result);
}

TEST_F(ModelWrapper_Test, StateDerivative_2) {

    state << 1, 2, 3, 4;
    control << 5;
    result = ddp_model->f(state, control);
    model.computeStateDeriv(state, control, known_result);

    ASSERT_EQ(known_result, result);
}