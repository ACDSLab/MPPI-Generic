//
// Created by mgandhi3 on 10/4/19.
//

#include <gtest/gtest.h>
#include <device_launch_parameters.h>
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

TEST(CartPole, SetGetParamsHost) {
    auto params = CartpoleParams(2.0, 3.0, 4.0);
    auto CP = Cartpole(0.1, 1, 1, 2);

    CP.setParams(params);
    auto CP_params = CP.getParams();

    EXPECT_FLOAT_EQ(params.cart_mass, CP_params.cart_mass);
    EXPECT_FLOAT_EQ(params.pole_mass, CP_params.pole_mass);
    EXPECT_FLOAT_EQ(params.pole_length, CP_params.pole_length);
}

__global__ void ParameterTestKernel(Cartpole CP) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Entering the kernel!\n");
    printf("The thread id is: %i\n", tid);
    if (tid == 0) {
        printf("The cart mass is: %f\n", *CP.cart_mass_d_);
    }
}
/*
 * This pattern works because we are passing the CartPole object via copy to the kernel, then the kernel invokes the
 * constructor of CartPole and copies the values into the device memory. Thus, the resulting device code has pointers
 * to the correct locations, but we are allocating twice as much memory as we need (I think...). Either way, this is bad
 */
TEST(CartPole, GetParamsDevice) {
    auto CP = Cartpole(0.1, 1, 1, 2);
    auto params = CartpoleParams(2.0, 3.0, 4.0);
    CP.setParams(params);

    ParameterTestKernel<<<1,1>>>(CP);
    cudaDeviceSynchronize();
    CudaCheckError();
    // Test kernel that will get the parameters from the device itself?
    FAIL();

}

TEST(CartPole, GetParamsDeviceStream) {
    FAIL();
}
