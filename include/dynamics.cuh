/*
Header file for dynamics
*/

#include <Eigen/Dense>
#include <stdio.h>
#include <math.h>

template<int S_DIM, int C_DIM>
class Dynamics
{
public:
  static const int STATE_DIM = S_DIM;
  static const int CONTROL_DIM = C_DIM;
  Eigen::Matrix<float, STATE_DIM, 1> state_der_;
  float dt;

  Dynamics();
  ~Dynamics();

  void xDot(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &state_der); //passing values in by reference
  void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control); //compute the Jacobians with respect to state and control
  void loadParams(); //figure out what to pass in here
  void paramsToDevice();
  void freeCudaMem();
  void printState();
  void printParams();

  __device__ void xDot(float* state, float* control, float* state_der);
}
