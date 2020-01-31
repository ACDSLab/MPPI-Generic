#include <gtest/gtest.h>
#include <mppi_core/normexp_kernel_test.cuh>
#include <utils/test_helper.h>
#include <random>
#include <algorithm>

TEST(NormExpKernel, computeBaselineCost_Test) {
    std::default_random_engine generator(7.0);
    std::normal_distribution<float> distribution(100.0,2.0);

    const int num_rollouts = 4196;
    std::array<float, num_rollouts> cost_vec = {0};

    // Use a range based for loop to set the cost
    for(auto& cost: cost_vec) {
        cost = distribution(generator);
    }

    float min_cost_known = *std::min_element(cost_vec.begin(), cost_vec.end());
    float min_cost_compute = mppi_common::computeBaselineCost(cost_vec.data(), num_rollouts);

    ASSERT_FLOAT_EQ(min_cost_compute, min_cost_known);
}

TEST(NormExpKernel, computeNormalizer_Test) {
    GTEST_SKIP() << "Not implemented";
}