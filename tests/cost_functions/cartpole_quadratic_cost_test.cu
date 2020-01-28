#include <gtest/gtest.h>
#include <cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <cost_functions/cartpole/cartpole_quadratic_cost_kernel_test.cuh>

TEST(CartPoleQuadraticCost, Constructor) {
    CartPoleQuadraticCost cost;

    // Test passes if the object is constructed without runetime error.
}

TEST(CartPoleQuadraticCost, BindStream) {
    cudaStream_t stream;
    HANDLE_ERROR(cudaStreamCreate(&stream));
    CartPoleQuadraticCost cost(stream);
    EXPECT_EQ(cost.stream_, stream) << "Stream binding failure.";
    HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(CartPoleQuadraticCost, GPUMemoryNull) {
    CartPoleQuadraticCost cost;
    ASSERT_EQ(cost.cost_d_, nullptr);
}

TEST(CartPoleQuadraticCost, GPUSetup) {
    CartPoleQuadraticCost cost;
    cost.GPUSetup();

    ASSERT_FALSE(cost.cost_d_ == nullptr);
}

TEST(CartPoleQuadraticCost, SetParamsCPU) {
    CartPoleQuadraticCost::CartPoleQuadraticCostParams new_params;
    new_params.cart_position_coeff = 3;
    new_params.pole_angle_coeff = 3;
    new_params.cart_velocity_coeff = 3;
    new_params.pole_angular_velocity_coeff = 3;
    new_params.control_force_coeff = 5;

    CartPoleQuadraticCost cost;

    cost.setParams(new_params);
    auto current_params = cost.getParams();

    EXPECT_FLOAT_EQ(new_params.cart_position_coeff, current_params.cart_position_coeff);
    EXPECT_FLOAT_EQ(new_params.pole_angle_coeff, current_params.pole_angle_coeff);
    EXPECT_FLOAT_EQ(new_params.cart_velocity_coeff, current_params.cart_velocity_coeff);
    EXPECT_FLOAT_EQ(new_params.pole_angular_velocity_coeff, current_params.pole_angular_velocity_coeff);
    EXPECT_FLOAT_EQ(new_params.control_force_coeff, current_params.control_force_coeff);

}

TEST(CartPoleQuadraticCost, SetParamsGPU) {
    CartPoleQuadraticCost cost;
    cost.GPUSetup();

    CartPoleQuadraticCost::CartPoleQuadraticCostParams new_params;
    new_params.cart_position_coeff = 5;
    new_params.pole_angle_coeff = 6;
    new_params.cart_velocity_coeff = 7;
    new_params.pole_angular_velocity_coeff = 8;
    new_params.control_force_coeff = 9;

    CartPoleQuadraticCost::CartPoleQuadraticCostParams gpu_params;

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

