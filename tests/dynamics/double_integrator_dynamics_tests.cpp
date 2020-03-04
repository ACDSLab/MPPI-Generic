//
// Created by mgandhi3 on 3/4/20.
//
#include <gtest/gtest.h>
#include <dynamics/double_integrator/di_dynamics.cuh>
#include <cuda_runtime.h>

TEST(DI_Dynamics, Construction) {
  auto model = new DoubleIntegratorDynamics();

  delete(model);
  FAIL();
}