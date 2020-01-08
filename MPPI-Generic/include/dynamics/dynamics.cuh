#pragma once

#ifndef DYNAMICS_CUH_
#define DYNAMICS_CUH_

/*
Header file for dynamics
*/

#include <Eigen/Dense>
#include <stdio.h>
#include <math.h>
#include <utils/managed.cuh>

template<int S_DIM, int C_DIM>
class Dynamics : public Managed
{
public:
  static const int STATE_DIM = S_DIM;
  static const int CONTROL_DIM = C_DIM;
  // Eigen::Matrix<float, STATE_DIM, 1> state_der_;
  float dt;

  Dynamics() = default;
  ~Dynamics() = default;

  void xDot(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &state_der); //passing values in by reference
  void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control); //compute the Jacobians with respect to state and control
  void loadParams(); //figure out what to pass in here
  void paramsToDevice();
  void freeCudaMem();
  void printState();
  void printParams();

  __device__ void xDot(float* state, float* control, float* state_der);
};

template <int S_DIM, int C_DIM>
const int Dynamics<S_DIM, C_DIM>::STATE_DIM;

template <int S_DIM, int C_DIM>
const int Dynamics<S_DIM, C_DIM>::CONTROL_DIM;
#endif // DYNAMICS_CUH_
