/*
Header file for dynamics
*/

#include <Eigen/Dense>

template<int S_DIM, int C_DIM>
class Dynamics
{
public:
  static const int STATE_DIM = S_DIM;
  static const int CONTROL_DIM = C_DIM;
  float dt;

  Dynamics();
  ~Dynamics();

  void xDot(Eigen::MatrixXf &state, Eigen::MatrixXf &control); //passing values in by reference
  void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control); //compute the Jacobians with respect to state and control
  void loadParams(); //figure out what to pass in here
  void paramsToDevice();
  void freeCudaMem();
  void printState();
  void printParams();
  void incrementState(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  __device__ void xDot(float* state, float* control);
  __device__ void computeGrad(float* state, float* control);
  __device__ void incrementState(float* state, float* control);
}
