//
// Created by jason on 1/8/20.
//

#ifndef MPPIGENERIC_AR_STANDARD_COST_KERNEL_TEST_CUH
#define MPPIGENERIC_AR_STANDARD_COST_KERNEL_TEST_CUH


__global__ void parameterTestKernel(ARStandardCost* cost, ARStandardCost::ARStandardCostParams& params, int& width, int& height);

void launchParameterTestKernel(const ARStandardCost& cost, ARStandardCost::ARStandardCostParams& params, int& width, int& height);

void launchCheckCudaArray(std::vector<float4>& result_arr, cudaArray* array, int number);

__global__ void checkCudaArrayKernel(float4* result_arr, cudaArray* array, int number);

void launchTransformTestKernel(std::vector<float3>& result, const ARStandardCost& cost);

__global__ void transformTestKernel(float3* results, ARStandardCost* cost);

void launchTextureTestKernel(const ARStandardCost& cost, std::vector<float4>& test_results, std::vector<float2>& test_indexes);

__global__ void textureTestKernel(ARStandardCost& cost, float4* test_results, float2* test_indexes, int num_points);

#endif //MPPIGENERIC_AR_STANDARD_COST_KERNEL_TEST_CUH
