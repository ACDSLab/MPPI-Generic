//
// Created by mgandhi on 5/23/20.
//

#ifndef MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
#define MPPIGENERIC_RMPPI_KERNEL_TEST_CUH

#include <mppi/core/mppi_common.cuh>
#include <curand.h>
#include <Eigen/Dense>



#if __CUDACC__
#include "rmppi_kernel_test.cu"
#endif

#endif //MPPIGENERIC_RMPPI_KERNEL_TEST_CUH
