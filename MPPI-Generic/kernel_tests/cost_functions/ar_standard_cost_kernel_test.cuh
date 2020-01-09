//
// Created by jason on 1/8/20.
//

#ifndef MPPIGENERIC_AR_STANDARD_COST_KERNEL_TEST_CUH
#define MPPIGENERIC_AR_STANDARD_COST_KERNEL_TEST_CUH


__global__ void parameterTestKernel(ARStandardCost& cost, float& desired_speed, int& num_timesteps,
                                    float3& r_c1, int& width, int& height);

void launchParameterTestKernel(const ARStandardCost& cost, float& desired_speed, int& num_timesteps,
                               float3& r_c1, int& width, int& height);

#endif //MPPIGENERIC_AR_STANDARD_COST_KERNEL_TEST_CUH
