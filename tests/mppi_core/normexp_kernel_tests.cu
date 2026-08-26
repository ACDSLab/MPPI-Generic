#include <gtest/gtest.h>
#include <kernel_tests/core/normexp_kernel_test.cuh>
#include <mppi/utils/test_helper.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

class NormExpKernel : public testing::Test
{
protected:
  void SetUp() override
  {
    generator = std::default_random_engine(7.0);
    distribution = std::normal_distribution<float>(100.0, 2.0);
  }

  void TearDown() override
  {
  }

  std::default_random_engine generator;
  std::normal_distribution<float> distribution;
};

template <int NUM_ROLLOUTS>
__global__ void computeNormalizerKernel(const float* __restrict__ costs, float* __restrict__ output)
{
  __shared__ float reduction_buffer[NUM_ROLLOUTS];
  int global_idx = threadIdx.x;
  int global_step = blockDim.x;
  float result = mppi::kernels::computeNormalizer(NUM_ROLLOUTS, costs, reduction_buffer, global_idx, global_step);
  if (threadIdx.x == 0)
  {
    *output = result;
  }
};

template <int NUM_ROLLOUTS>
__global__ void computeBaselineCostKernel(const float* __restrict__ costs, float* __restrict__ output)
{
  __shared__ float reduction_buffer[NUM_ROLLOUTS];
  int global_idx = threadIdx.x;
  int global_step = blockDim.x;
  float result = mppi::kernels::computeBaselineCost(NUM_ROLLOUTS, costs, reduction_buffer, global_idx, global_step);
  if (threadIdx.x == 0)
  {
    *output = result;
  }
};

TEST_F(NormExpKernel, computeBaselineCost_Test)
{
  const int num_rollouts = 4196;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float min_cost_known = *std::min_element(cost_vec.begin(), cost_vec.end());
  float min_cost_compute = mppi::kernels::computeBaselineCost(cost_vec.data(), num_rollouts);

  ASSERT_FLOAT_EQ(min_cost_compute, min_cost_known);
}

TEST_F(NormExpKernel, computeNormalizer_Test)
{
  const int num_rollouts = 1024;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float sum_cost_known = std::accumulate(cost_vec.begin(), cost_vec.end(), 0.0);
  float sum_cost_compute = mppi::kernels::computeNormalizer(cost_vec.data(), num_rollouts);

  ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

TEST_F(NormExpKernel, computeNormalizerDevice_Test)
{
  const int num_rollouts = 6048;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (int i = 0; i < cost_vec.size(); i++)
  {
    cost_vec[i] = distribution(generator);
  }
  float* norm_d;
  float* costs_d;
  float sum_cost_compute;
  HANDLE_ERROR(cudaMalloc((void**)&norm_d, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMemcpy(costs_d, cost_vec.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));
  computeNormalizerKernel<num_rollouts><<<1, 1024>>>(costs_d, norm_d);
  HANDLE_ERROR(cudaMemcpy(&sum_cost_compute, norm_d, sizeof(float), cudaMemcpyDeviceToHost));

  float sum_cost_known = std::accumulate(cost_vec.begin(), cost_vec.end(), 0.0);
  ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

TEST_F(NormExpKernel, computeBaselineCostDevice_Test)
{
  const int num_rollouts = 6048;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (int i = 0; i < cost_vec.size(); i++)
  {
    cost_vec[i] = cost_vec.size() - i;
  }
  std::cout << std::endl;
  float* norm_d;
  float* costs_d;
  float sum_cost_compute;
  HANDLE_ERROR(cudaMalloc((void**)&norm_d, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMemcpy(costs_d, cost_vec.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));
  computeBaselineCostKernel<num_rollouts><<<1, 1024>>>(costs_d, norm_d);
  HANDLE_ERROR(cudaMemcpy(&sum_cost_compute, norm_d, sizeof(float), cudaMemcpyDeviceToHost));

  float sum_cost_known = *std::min_element(cost_vec.begin(), cost_vec.end());
  ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

template <int NUM_ROLLOUTS>
void checkBaselineCostDeviceOdd()
{
  // Odd rollout counts exercise the leftover-element fixup in the first
  // reduction stage; costs descend so the true minimum sits at the last
  // index, the exact element the fixup is responsible for.
  std::array<float, NUM_ROLLOUTS> cost_vec = { 0 };
  for (int i = 0; i < cost_vec.size(); i++)
  {
    cost_vec[i] = cost_vec.size() - i;
  }
  float* min_d;
  float* costs_d;
  float min_cost_compute;
  HANDLE_ERROR(cudaMalloc((void**)&min_d, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * NUM_ROLLOUTS));
  HANDLE_ERROR(cudaMemcpy(costs_d, cost_vec.data(), sizeof(float) * NUM_ROLLOUTS, cudaMemcpyHostToDevice));
  computeBaselineCostKernel<NUM_ROLLOUTS><<<1, 1024>>>(costs_d, min_d);
  HANDLE_ERROR(cudaMemcpy(&min_cost_compute, min_d, sizeof(float), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(min_d));
  HANDLE_ERROR(cudaFree(costs_d));

  float min_cost_known = *std::min_element(cost_vec.begin(), cost_vec.end());
  ASSERT_FLOAT_EQ(min_cost_compute, min_cost_known);
}

TEST_F(NormExpKernel, computeBaselineCostDeviceOddRollouts_Test)
{
  checkBaselineCostDeviceOdd<999>();   // odd, below the 1024-thread launch
  checkBaselineCostDeviceOdd<6049>();  // odd, above it (strided loops multi-trip)
}

template <int NUM_ROLLOUTS>
void checkNormalizerDeviceOdd()
{
  // Small exactly-representable integer values keep every partial sum exact
  // in float, so the serial reference and the parallel tree sum are
  // bit-identical and the exact-equality assert is independent of
  // reduction order.
  std::array<float, NUM_ROLLOUTS> cost_vec = { 0 };
  for (int i = 0; i < cost_vec.size(); i++)
  {
    cost_vec[i] = (i % 7) + 1;
  }
  float* norm_d;
  float* costs_d;
  float sum_cost_compute;
  HANDLE_ERROR(cudaMalloc((void**)&norm_d, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * NUM_ROLLOUTS));
  HANDLE_ERROR(cudaMemcpy(costs_d, cost_vec.data(), sizeof(float) * NUM_ROLLOUTS, cudaMemcpyHostToDevice));
  computeNormalizerKernel<NUM_ROLLOUTS><<<1, 1024>>>(costs_d, norm_d);
  HANDLE_ERROR(cudaMemcpy(&sum_cost_compute, norm_d, sizeof(float), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(norm_d));
  HANDLE_ERROR(cudaFree(costs_d));

  float sum_cost_known = std::accumulate(cost_vec.begin(), cost_vec.end(), 0.0f);
  ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

TEST_F(NormExpKernel, computeNormalizerDeviceOddRollouts_Test)
{
  checkNormalizerDeviceOdd<999>();   // odd, below the 1024-thread launch
  checkNormalizerDeviceOdd<6049>();  // odd, above it (strided loops multi-trip)
}

TEST_F(NormExpKernel, fullGPUcomputeWeightsEveryIteration_Test)
{
  // Focused regression for the qualifier-dependent reduction failure: the
  // defect is data-dependent, so this samples fresh costs each iteration and
  // checks EVERY iteration (the timing comparison test only checks the
  // final one). No timing, stops at the first bad dataset.
  const int num_rollouts = 10000;
  const int blocksize_x = 8;
  const int num_iterations = 500;
  std::array<float, num_rollouts> cost_vec = { 0 };
  std::array<float, num_rollouts> host_dev_costs = { 0 };
  std::array<float, num_rollouts> dev_only_costs = { 0 };
  float lambda = 0.3;
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  float* costs_dev_only_d;
  float* costs_host_only_d;
  float2* baseline_and_normalizer_d;
  float2 host_components, device_components;
  HANDLE_ERROR(cudaMalloc((void**)&baseline_and_normalizer_d, sizeof(float2)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_dev_only_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&costs_host_only_d, sizeof(float) * num_rollouts));

  for (int iter = 0; iter < num_iterations; iter++)
  {
    for (auto& cost : cost_vec)
    {
      cost = distribution(generator);
    }
    HANDLE_ERROR(
        cudaMemcpyAsync(costs_dev_only_d, cost_vec.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(costs_host_only_d, cost_vec.data(), sizeof(float) * num_rollouts,
                                 cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    host_components.x = mppi::kernels::computeBaselineCost(cost_vec.data(), num_rollouts);
    mppi::kernels::launchNormExpKernel(num_rollouts, blocksize_x, costs_host_only_d, 1.0 / lambda, host_components.x,
                                       stream, false);
    HANDLE_ERROR(cudaMemcpyAsync(host_dev_costs.data(), costs_host_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    host_components.y = mppi::kernels::computeNormalizer(host_dev_costs.data(), num_rollouts);

    mppi::kernels::launchWeightTransformKernel<num_rollouts>(costs_dev_only_d, baseline_and_normalizer_d, 1.0 / lambda,
                                                             1, stream, false);
    HANDLE_ERROR(cudaMemcpyAsync(dev_only_costs.data(), costs_dev_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(
        cudaMemcpyAsync(&device_components, baseline_and_normalizer_d, sizeof(float2), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    SCOPED_TRACE("iteration " + std::to_string(iter));
    EXPECT_FLOAT_EQ(device_components.x, host_components.x);
    EXPECT_FLOAT_EQ(device_components.y, host_components.y);
    // The two weight arrays should be bit-identical (same device transform on
    // the same baseline), so scan for the first mismatch and report just that
    // pair instead of running num_rollouts assertions per iteration.
    for (int i = 0; i < num_rollouts; i++)
    {
      if (dev_only_costs[i] != host_dev_costs[i])
      {
        EXPECT_FLOAT_EQ(dev_only_costs[i], host_dev_costs[i]) << "first weight mismatch at index " << i;
        break;
      }
    }
    if (::testing::Test::HasFailure())
    {
      break;
    }
  }
  HANDLE_ERROR(cudaFree(baseline_and_normalizer_d));
  HANDLE_ERROR(cudaFree(costs_dev_only_d));
  HANDLE_ERROR(cudaFree(costs_host_only_d));
  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST_F(NormExpKernel, computeExpNorm_Test)
{
  const int num_rollouts = 555;
  std::array<float, num_rollouts> cost_vec = { 0 };
  std::array<float, num_rollouts> normalized_compute = { 0 };
  std::array<float, num_rollouts> normalized_known = { 0 };
  float gamma = 0.3;

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float baseline = *std::min_element(cost_vec.begin(), cost_vec.end());

  for (int i = 0; i < num_rollouts; i++)
  {
    normalized_known[i] = expf(-gamma * (cost_vec[i] - baseline));
  }

  launchNormExp_KernelTest<num_rollouts>(cost_vec, gamma, baseline, normalized_compute);

  array_assert_float_eq<num_rollouts>(normalized_compute, normalized_known);
}

TEST_F(NormExpKernel, comparisonTestAutorallyMPPI_Generic)
{
  const int num_rollouts = 28754;
  const int blocksize_x = 8;
  const int blocksize_y = 8;
  std::array<float, num_rollouts> cost_vec = { 0 };
  std::array<float, num_rollouts> normalized_autorally = { 0 };
  std::array<float, num_rollouts> normalized_generic = { 0 };
  float gamma = 0.3;

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float baseline = *std::min_element(cost_vec.begin(), cost_vec.end());

  launchGenericNormExpKernelTest<num_rollouts, blocksize_x>(cost_vec, gamma, baseline, normalized_generic);

  for (int i = 0; i < num_rollouts; i++)
  {
    float cost = cost_vec[i] - baseline;
    cost = expf(-gamma * cost);
    EXPECT_FLOAT_EQ(normalized_generic[i], cost);
  }
}

TEST_F(NormExpKernel, comparisonTestHostvsDeviceBaselineNormalizerCalculation)
{
  const int num_rollouts = 10000;
  const int blocksize_x = 8;
  const int num_iterations = 2500;
  std::array<float, num_rollouts> cost_vec = { 0 };
  std::array<float, num_rollouts> host_dev_costs = { 0 };
  std::array<float, num_rollouts> dev_only_costs = { 0 };
  float lambda = 0.3;
  double old_method_ms = 0;
  double new_method_ms = 0;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float* costs_dev_only_d;
  float* costs_host_only_d;
  float2* baseline_and_normalizer_d;
  float2 host_components, device_components;
  HANDLE_ERROR(cudaMalloc((void**)&baseline_and_normalizer_d, sizeof(float2)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_dev_only_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&costs_host_only_d, sizeof(float) * num_rollouts));

  // Use a range based for loop to set the cost
  for (int i = 0; i < num_iterations; i++)
  {
    for (auto& cost : cost_vec)
    {
      cost = distribution(generator);
    }

    /**
     * @brief Prep CUDA components
     *
     */
    HANDLE_ERROR(cudaMemcpyAsync(costs_dev_only_d, cost_vec.data(), sizeof(float) * num_rollouts,
                                 cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(costs_host_only_d, cost_vec.data(), sizeof(float) * num_rollouts,
                                 cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    auto start_old_method_t = std::chrono::steady_clock::now();
    // Run old method to transform costs
    HANDLE_ERROR(cudaMemcpyAsync(host_dev_costs.data(), costs_host_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    host_components.x = mppi::kernels::computeBaselineCost(host_dev_costs.data(), num_rollouts);
    mppi::kernels::launchNormExpKernel(num_rollouts, blocksize_x, costs_host_only_d, 1.0 / lambda, host_components.x,
                                       stream, false);
    HANDLE_ERROR(cudaMemcpyAsync(host_dev_costs.data(), costs_host_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    host_components.y = mppi::kernels::computeNormalizer(host_dev_costs.data(), num_rollouts);
    old_method_ms += (std::chrono::steady_clock::now() - start_old_method_t).count() / 1e6;

    auto start_new_method_t = std::chrono::steady_clock::now();
    // Run new method to transform costs
    mppi::kernels::launchWeightTransformKernel<num_rollouts>(costs_dev_only_d, baseline_and_normalizer_d, 1.0 / lambda,
                                                             1, stream, false);
    HANDLE_ERROR(cudaMemcpyAsync(dev_only_costs.data(), costs_dev_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(
        cudaMemcpyAsync(&device_components, baseline_and_normalizer_d, sizeof(float2), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    new_method_ms += (std::chrono::steady_clock::now() - start_new_method_t).count() / 1e6;
  }

  std::cout << "Old method averaged " << old_method_ms / num_iterations << " ms and the new method averaged "
            << new_method_ms / num_iterations << " ms" << std::endl;

  for (int i = 0; i < num_rollouts; i++)
  {
    ASSERT_FLOAT_EQ(dev_only_costs[i], host_dev_costs[i]);
  }
  ASSERT_FLOAT_EQ(device_components.x, host_components.x);
  ASSERT_FLOAT_EQ(device_components.y, host_components.y);
}
