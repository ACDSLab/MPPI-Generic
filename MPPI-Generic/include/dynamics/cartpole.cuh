#include "dynamics.cuh"

template<int S_DIM, int, C_DIM>

class Cartpole: public Dynamics
{
public:
  float cart_mass_;
  float pole_mass_;
  float pole_length_;
  const float gravity_ = 9.81;

  //CUDA coefficients
  float* cart_mass_d_;
  float* pole_mass_d_;
  float* pole_length_d_;

  Cartpole(float delta_t, float cart_mass, float pole_mass, float pole_length);
  ~Cartpole();

  void xDot(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &state_der);  //passing values in by reference
  void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &A, Eigen::MatrixXf &B); //compute the Jacobians with respect to state and control
  void loadParams(); //figure out what to pass in here
  void paramsToDevice();
  void freeCudaMem();
  void printState();
  void printParams();

  __device__ void xDot(float* state, float* control, float* state_der);
}
