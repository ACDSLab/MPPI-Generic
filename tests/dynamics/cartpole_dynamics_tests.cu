//
// Created by mgandhi3 on 10/4/19.
//

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <dynamics/cartpole/cartpole_kernel_test.cuh>
#include <cuda_runtime.h>

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

TEST(CartPole, SetGetParamsHost) {
    auto params = CartpoleParams(2.0, 3.0, 4.0);
    auto CP = Cartpole(0.1, 1, 1, 2);

    CP.setParams(params);
    auto CP_params = CP.getParams();

    EXPECT_FLOAT_EQ(params.cart_mass, CP_params.cart_mass);
    EXPECT_FLOAT_EQ(params.pole_mass, CP_params.pole_mass);
    EXPECT_FLOAT_EQ(params.pole_length, CP_params.pole_length);
}


TEST(CartPole, CartPole_GPUSetup_Test) {
    auto CP_host = new Cartpole(0.1, 1, 2, 2);
    CP_host->GPUSetup();
    // float mass;
    // launchParameterTestKernel(*CP_host, mass);

    EXPECT_TRUE(CP_host->GPUMemStatus_);

    delete(CP_host);
}

TEST(CartPole, GetCartMassFromGPU) {
    auto CP_host = new Cartpole(0.1, 1, 1, 2);
    CP_host->GPUSetup();

    auto params = CartpoleParams(2.0, 3.0, 4.0);
    CP_host->setParams(params);
    float mass;

    launchCartMassTestKernel(*CP_host, mass);

    EXPECT_FLOAT_EQ(params.cart_mass, mass);

    CP_host->freeCudaMem();
    delete(CP_host);
}

TEST(CartPole, GetPoleMassFromGPU) {
    auto CP_host = new Cartpole(0.1, 1, 1, 2);
    CP_host->GPUSetup();

    auto params = CartpoleParams(2.0, 3.0, 4.0);
    CP_host->setParams(params);
    float mass;

    launchPoleMassTestKernel(*CP_host, mass);

    EXPECT_FLOAT_EQ(params.pole_mass, mass);

    CP_host->freeCudaMem();
    delete(CP_host);
}

TEST(CartPole, GetPoleLengthFromGPU) {
    auto CP_host = new Cartpole(0.1, 1, 1, 2);
    CP_host->GPUSetup();

    auto params = CartpoleParams(2.0, 3.0, 4.0);
    CP_host->setParams(params);
    float length;

    launchPoleLengthTestKernel(*CP_host, length);

    EXPECT_FLOAT_EQ(params.pole_length, length);

    CP_host->freeCudaMem();
    delete(CP_host);
}

TEST(CartPole, GetGravityFromGPU) {
    auto CP_host = new Cartpole(0.1, 1, 1, 2);
    CP_host->GPUSetup();

    auto params = CartpoleParams(2.0, 3.0, 4.0);
    CP_host->setParams(params);
    float gravity_gpu;

    launchGravityTestKernel(*CP_host, gravity_gpu);

    EXPECT_FLOAT_EQ(CP_host->getGravity(), gravity_gpu);

    CP_host->freeCudaMem();
    delete(CP_host);
}

TEST(CartPole, TestDynamicsGPU) {
    auto CP_host = new Cartpole(0.1, 1, 1, 2);
    CP_host->GPUSetup();

    auto params = CartpoleParams(2.0, 3.0, 4.0);
    CP_host->setParams(params);

    Eigen::MatrixXf state = Eigen::MatrixXf::Zero(Cartpole::STATE_DIM,1);
    state(0) = 0.1;
    state(1) = 0.3;
    state(2) = 0.23;
    state(3) = 0.334;
    Eigen::MatrixXf control = Eigen::MatrixXf::Ones(Cartpole::CONTROL_DIM, 1);

    // These variables will be changed so initialized to the right size only
    Eigen::MatrixXf state_der_cpu = Eigen::MatrixXf::Zero(Cartpole::STATE_DIM,1);

    float state_der_gpu[Cartpole::STATE_DIM];

    // Run dynamics on CPU
    CP_host->xDot(state, control, state_der_cpu);
    // Run dynamics on GPU
    launchDynamicsTestKernel(*CP_host, state.data(), control.data(), state_der_gpu);

    // Compare CPU and GPU Results
    for (int i = 0; i < Cartpole::STATE_DIM; i++) {
        EXPECT_FLOAT_EQ(state_der_cpu(i), state_der_gpu[i]);
    }

    CP_host->freeCudaMem();
    delete(CP_host);
}
