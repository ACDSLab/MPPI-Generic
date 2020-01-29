#include <gtest/gtest.h>
#include <cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <cost_functions/cartpole/cartpole_quadratic_cost_kernel_test.cuh>
#include <array>

TEST(CartPoleQuadraticCost, Constructor) {
    CartpoleQuadraticCost cost;

    // Test passes if the object is constructed without runetime error.
}

TEST(CartPoleQuadraticCost, BindStream) {
    cudaStream_t stream;
    HANDLE_ERROR(cudaStreamCreate(&stream));
    CartpoleQuadraticCost cost(stream);
    EXPECT_EQ(cost.stream_, stream) << "Stream binding failure.";
    HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(CartPoleQuadraticCost, GPUMemoryNull) {
    CartpoleQuadraticCost cost;
    ASSERT_EQ(cost.cost_d_, nullptr);
}

TEST(CartPoleQuadraticCost, GPUSetup) {
    CartpoleQuadraticCost cost;
    cost.GPUSetup();

    ASSERT_FALSE(cost.cost_d_ == nullptr);
}

TEST(CartPoleQuadraticCost, SetParamsCPU) {
    CartpoleQuadraticCost::CartpoleQuadraticCostParams new_params;
    new_params.cart_position_coeff = 3;
    new_params.pole_angle_coeff = 3;
    new_params.cart_velocity_coeff = 3;
    new_params.pole_angular_velocity_coeff = 3;
    new_params.control_force_coeff = 5;

    CartpoleQuadraticCost cost;

    cost.setParams(new_params);
    auto current_params = cost.getParams();

    EXPECT_FLOAT_EQ(new_params.cart_position_coeff, current_params.cart_position_coeff);
    EXPECT_FLOAT_EQ(new_params.pole_angle_coeff, current_params.pole_angle_coeff);
    EXPECT_FLOAT_EQ(new_params.cart_velocity_coeff, current_params.cart_velocity_coeff);
    EXPECT_FLOAT_EQ(new_params.pole_angular_velocity_coeff, current_params.pole_angular_velocity_coeff);
    EXPECT_FLOAT_EQ(new_params.control_force_coeff, current_params.control_force_coeff);

}

TEST(CartPoleQuadraticCost, SetParamsGPU) {
    CartpoleQuadraticCost cost;
    cost.GPUSetup();

    CartpoleQuadraticCost::CartpoleQuadraticCostParams new_params;
    new_params.cart_position_coeff = 5;
    new_params.pole_angle_coeff = 6;
    new_params.cart_velocity_coeff = 7;
    new_params.pole_angular_velocity_coeff = 8;
    new_params.control_force_coeff = 9;

    CartpoleQuadraticCost::CartpoleQuadraticCostParams gpu_params;

    cost.setParams(new_params);

    if (cost.GPUMemStatus_) {
        // Launch kernel to grab parameters from the GPU
        launchParameterTestKernel(cost, gpu_params);
    } else {
        FAIL() << "GPU Setup has not been called or is not functioning.";
    }

    EXPECT_FLOAT_EQ(new_params.cart_position_coeff, gpu_params.cart_position_coeff);
    EXPECT_FLOAT_EQ(new_params.pole_angle_coeff, gpu_params.pole_angle_coeff);
    EXPECT_FLOAT_EQ(new_params.cart_velocity_coeff, gpu_params.cart_velocity_coeff);
    EXPECT_FLOAT_EQ(new_params.pole_angular_velocity_coeff, gpu_params.pole_angular_velocity_coeff);
    EXPECT_FLOAT_EQ(new_params.control_force_coeff, gpu_params.control_force_coeff);
}

TEST(CartPoleQuadraticCost, ComputeStateCost) {
    CartpoleQuadraticCost cost;

    std::array<float, 4> state = {1.f, 2.f, 3.f, 4.f};

    float cost_compute = cost.getStateCost(state.data());
    float cost_known = state[0]*state[0]*cost.getParams().cart_position_coeff +
            state[1]*state[1]*cost.getParams().cart_velocity_coeff +
            state[2]*state[2]*cost.getParams().pole_angle_coeff +
            state[3]*state[3]*cost.getParams().pole_angular_velocity_coeff;

    ASSERT_EQ(cost_known, cost_compute);
}

TEST(CartPoleQuadraticCost, ComputeControlCost) {
    CartpoleQuadraticCost cost;

    float u = 10;
    float du = 0.4;
    float var = 1;

    float cost_compute = cost.getControlCost(&u, &du, &var);
    float cost_known = cost.getParams().control_force_coeff*du*(u - du) / (var*var);
    ASSERT_EQ(cost_known, cost_compute);
}

TEST(CartPoleQuadraticCost, ComputeRunningCost) {
    CartpoleQuadraticCost cost;

    std::array<float, 4> state = {5.f, 3.f, 2.f, 4.f};
    float u = 6;
    float du = 0.3;
    float var = 2;

    float cost_compute = cost.computeRunningCost(state.data(), &u, &du, &var);
    float cost_known = state[0]*state[0]*cost.getParams().cart_position_coeff +
                       state[1]*state[1]*cost.getParams().cart_velocity_coeff +
                       state[2]*state[2]*cost.getParams().pole_angle_coeff +
                       state[3]*state[3]*cost.getParams().pole_angular_velocity_coeff+
                       cost.getParams().control_force_coeff*du*(u - du) / (var*var);
    ASSERT_EQ(cost_known, cost_compute);
}

