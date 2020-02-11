//
// Created by jgibson37 on 2/7/20.
//

#ifndef MPPIGENERIC_AR_ROBUST_COST_KERNEL_TEST_CUH
#define MPPIGENERIC_AR_ROBUST_COST_KERNEL_TEST_CUH

#include "ar_standard_cost_kernel_test.cuh"

__global__ void getCostmapCostTestKernel(ARRobustCost<>* cost, float* test_xu, float* cost_results, int num_points);

void launchGetCostmapCostTestKernel(ARRobustCost<>& cost, std::vector<std::array<float, 9>>& test_xu, std::vector<float>& cost_results);

#include "ar_robust_cost_kernel_test.cu"

#endif //MPPIGENERIC_AR_ROBUST_COST_KERNEL_TEST_CUH

