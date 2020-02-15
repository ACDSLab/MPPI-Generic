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
  float dt_;


  Dynamics() = default;
  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a Dynamics reference
   */
  ~Dynamics() = default;

  /**
   * runs dynamics using state and control and sets it to state
   * derivative. Everything is Eigen Matrices, not Eigen Vectors!
   *
   * @param state     input of current state, passed by reference
   * @param control   input of currrent control, passed by reference
   * @param state_der output of new state derivative, passed by reference
   */
  void xDot(Eigen::MatrixXf &state,
                    Eigen::MatrixXf &control,
                    Eigen::MatrixXf &state_der);

  /**
   *
   * @param state
   * @param control
   */
  void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control); //compute the Jacobians with respect to state and control

  /**
   *
   */
  void loadParams(); //figure out what to pass in here

  /**
   *
   */
  void paramsToDevice();

  /**
   *
   */
  void freeCudaMem();

  /**
   *
   */
  void printState();

  /**
   *
   */
  void printParams();

  /**
   * TODO: Replace with thrust::array<float, STATE_SIZE>
   * to ensure that sizes are enforced
   *
   * Can't make this virtual!!
   * Making this a pure virtual function with:
   *
   * virtual ... = 0;
   *
   * makes the GPU version of this function an illegal
   * memory access so this must be left as follows for now.
   *
   * Even just making it a virtual function causes problems for executables
   * as there is no definition of this virtual function for the executable
   * to link to. Defining it to an empty function:
   * virtual ... = {}
   * causes the same problem as a pure virtual function
   */
  __device__ void xDot(float* state,
                                float* control,
                                float* state_der);
};

template <int S_DIM, int C_DIM>
const int Dynamics<S_DIM, C_DIM>::STATE_DIM;

template <int S_DIM, int C_DIM>
const int Dynamics<S_DIM, C_DIM>::CONTROL_DIM;
#endif // DYNAMICS_CUH_
