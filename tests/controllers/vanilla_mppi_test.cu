#include <gtest/gtest.h>
#include <controllers/cartpole_mppi.cuh>

TEST(Cartpole_VanillaMPPI, BindToStream) {
    Cartpole model(0.1, 1.0, 1.0, 1.0);
    CartpoleQuadraticCost cost;

    float dt = 0.01;
    cudaStream_t stream;

    HANDLE_ERROR(cudaStreamCreate(&stream));

    auto CartpoleController = VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 150, 64, 8>(&model, &cost, dt, stream);

    EXPECT_EQ(CartpoleController.stream_, CartpoleController.model_->stream_)
                        << "Stream bind to dynamics failure";
    EXPECT_EQ(CartpoleController.stream_, CartpoleController.cost_->stream_)
                        << "Stream bind to cost failure";
    HANDLE_ERROR(cudaStreamDestroy(stream));

}
