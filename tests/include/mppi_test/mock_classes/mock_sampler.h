//
// Created by jason on 2/21/24.
//

#ifndef MPPIGENERIC_TESTS_INCLUDE_MPPI_TEST_MOCK_CLASSES_MOCK_SAMPLER_H_
#define MPPIGENERIC_TESTS_INCLUDE_MPPI_TEST_MOCK_CLASSES_MOCK_SAMPLER_H_

#include <mppi/sampling_distributions/sampling_distribution.cuh>
#include <mppi_test/mock_classes/mock_dynamics.h>

class MockSamplingDistribution
  : public mppi::sampling_distributions::SamplingDistribution<
        MockSamplingDistribution, mppi::sampling_distributions::SamplingParams, mockDynamicsParams>
{
  MOCK_METHOD1(bindToStream, void(cudaStream_t stream));
  MOCK_METHOD0(GPUSetup, void());
  MOCK_METHOD1(resizeVisualizationCotnrolTrajectories, void(bool synchronize));
  MOCK_METHOD1(allocateCUDAMemory, void(bool synchronize));
  MOCK_METHOD0(freeCudaMem, void());
};
#endif  // MPPIGENERIC_TESTS_INCLUDE_MPPI_TEST_MOCK_CLASSES_MOCK_SAMPLER_H_
