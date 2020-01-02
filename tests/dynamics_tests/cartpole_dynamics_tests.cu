//
// Created by mgandhi3 on 10/4/19.
//

#include <gtest/gtest.h>
#include "cartpole.cuh"

TEST(CartPole, StateDim) {
    auto CP = Cartpole(0.1, 1, 1, 1);
    EXPECT_EQ(4, Cartpole::STATE_DIM);
}

TEST(CartPole, ControlDim) {
    auto CP = Cartpole(0.1,1,1,1);
    EXPECT_EQ(1, Cartpole::CONTROL_DIM);
}

TEST(CartPole, Equilibrium) {
    auto CP = Cartpole(0.1,1,1,1);
    
    Eigen::MatrixXf state(Cartpole::STATE_DIM,1);
    state << 0,0,0,0;
    
    Eigen::MatrixXf control(Cartpole::CONTROL_DIM,1);
    control << 0;

    Eigen::MatrixXf state_dot_compute(Cartpole::STATE_DIM,1);
    state_dot_compute << 1,1,1,1;

    Eigen::MatrixXf state_dot_known(Cartpole::STATE_DIM,1);
    state_dot_known << 0,0,0,0;

    CP.xDot(state, control, state_dot_compute);
    for (int i = 0; i < Cartpole::STATE_DIM; i++) {
        EXPECT_NEAR(state_dot_known(i), state_dot_compute(i), 1e-4) << "Failed at index: " << i;
    }
}

TEST(CartPole, BindStream) {
    cudaStream_t stream;

    HANDLE_ERROR(cudaStreamCreate(&stream));

    auto CP = Cartpole(0.1, 1, 1, 2, stream);

    EXPECT_EQ(CP.stream_, stream) << "Stream binding failure.";

    HANDLE_ERROR(cudaStreamDestroy(stream));

}

