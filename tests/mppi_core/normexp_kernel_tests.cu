#include <gtest/gtest.h>
#include <mppi/core/normexp_kernel_test.cuh>
#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <numeric>


class NormExpKernel: public testing::Test {
protected:
    void SetUp() override {
        generator = std::default_random_engine(7.0);
        distribution = std::normal_distribution<float>(100.0,2.0);
    }

    void TearDown() override {

    }

    std::default_random_engine generator;
    std::normal_distribution<float> distribution;
};

TEST_F(NormExpKernel, computeBaselineCost_Test) {
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

TEST_F(NormExpKernel, computeNormalizer_Test) {
    const int num_rollouts = 1024;
    std::array<float, num_rollouts> cost_vec = {0};

    // Use a range based for loop to set the cost
    for(auto& cost: cost_vec) {
        cost = distribution(generator);
    }

    float sum_cost_known = std::accumulate(cost_vec.begin(), cost_vec.end(), 0.f);
    float sum_cost_compute = mppi_common::computeNormalizer(cost_vec.data(), num_rollouts);

    ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

TEST_F(NormExpKernel, computeExpNorm_Test) {
    const int num_rollouts = 555;
    std::array<float, num_rollouts> cost_vec = {0};
    std::array<float, num_rollouts> normalized_compute = {0};
    std::array<float, num_rollouts> normalized_known = {0};
    float gamma = 0.3;

    // Use a range based for loop to set the cost
    for(auto& cost: cost_vec) {
        cost = distribution(generator);
    }

    float baseline = *std::min_element(cost_vec.begin(), cost_vec.end());

    for (int i = 0; i < num_rollouts; i++) {
        normalized_known[i] = expf(-gamma*(cost_vec[i] - baseline));
    }

    launchNormExp_KernelTest<num_rollouts>(cost_vec, gamma, baseline, normalized_compute);

    array_assert_float_eq<num_rollouts>(normalized_compute, normalized_known);
}

TEST_F(NormExpKernel, comparisonTestAutorallyMPPI_Generic) {
    const int num_rollouts = 28754;
    const int blocksize_x = 8;
    const int blocksize_y = 8;
    std::array<float, num_rollouts> cost_vec = {0};
    std::array<float, num_rollouts> normalized_autorally = {0};
    std::array<float, num_rollouts> normalized_generic = {0};
    float gamma = 0.3;

    // Use a range based for loop to set the cost
    for(auto& cost: cost_vec) {
        cost = distribution(generator);
    }

    float baseline = *std::min_element(cost_vec.begin(), cost_vec.end());

    launchGenericNormExpKernelTest<num_rollouts, blocksize_x>
            (cost_vec, gamma, baseline, normalized_generic);

    launchAutorallyNormExpKernelTest<num_rollouts, blocksize_x, blocksize_y>
            (cost_vec, gamma, baseline, normalized_autorally);

    array_assert_float_eq<num_rollouts>(normalized_autorally, normalized_generic);
}