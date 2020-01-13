//
// Created by jgibson37 on 1/13/20.
//

#include <gtest/gtest.h>
#include <dynamics/autorally/ar_nn_model.cuh>
#include <dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>

TEST(ARNeuralNetDynamics, Constructor) {
  NeuralNetModel<4, 5, 6, 7> model;
}
