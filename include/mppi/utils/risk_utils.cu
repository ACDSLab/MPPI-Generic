#include <mppi/utils/risk_utils.cuh>

namespace mppi
{
__host__ __device__ float RiskMeasure::var(float* __restrict__ costs, const int num_costs, float alpha)
{
  float cost = 0;
#ifdef __CUDA_ARCH__
  thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(costs);
  thrust::sort(thrust::seq, thrust_ptr, thrust_ptr + num_costs);
#else
  std::sort(costs, costs + num_costs);
#endif
  float h_idx = h_index(num_costs, alpha);
  int next_idx = min((int)ceilf(h_idx), num_costs - 1);
  int prev_idx = max((int)floorf(h_idx), 0);
  cost = costs[prev_idx] + (h_idx - prev_idx) * (costs[next_idx] - costs[prev_idx]);
  return cost;
}

__host__ __device__ float RiskMeasure::cvar(float* __restrict__ costs, const int num_costs, float alpha)
{
  float cost = 0;
  float value_at_risk = var(costs, num_costs, alpha);  // also sorts costs
  int num_costs_above = 1;
  float sum_costs_above = value_at_risk;
  float h_idx = h_index(num_costs, alpha);
  for (int i = ceilf(h_idx); i < num_costs; i++)
  {
    num_costs_above++;
    sum_costs_above += costs[i];
  }
  cost = sum_costs_above / num_costs_above;
  return cost;
}
}  // namespace mppi
