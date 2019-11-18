#include <dynamics/cartpole.cuh>

Cartpole::Cartpole(float delta_t, float cart_mass, float pole_mass, float pole_length)
{
  cart_mass_ = cart_mass;
  pole_mass_ = pole_mass;
  pole_length_ = pole_length;
  dt_ = delta_t;

  cudaMalloc((void**)&cart_mass_d_, sizeof(int));
  cudaMalloc((void**)&pole_mass_d_, sizeof(int));
  cudaMalloc((void**)&pole_length_d_, sizeof(int));
}

void Cartpole::Cartpole::xDot(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &state_der)
{
  float theta = state(2);
  float theta_dot = state(3);
  float force = control(0);

  state_der(0) = state(1);
  state_der(1) = 1/(cart_mass_+pole_mass_*powf(sinf(theta),2.0))*(force+pole_mass_*sinf(theta)*(pole_length_*(powf(theta_dot,2.0))+gravity_*cosf(theta)));
  state_der(2) = state(3);
  state_der(3) = 1/(pole_length_*(cart_mass_+pole_mass_*(powf(sinf(theta),2.0))))*(-force*cosf(theta)-pole_mass_*pole_length_*(powf(theta_dot,2.0))*cosf(theta)*sinf(theta)-(cart_mass_+pole_mass_)*gravity_*sinf(theta));
}

void Cartpole::computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &A, Eigen::MatrixXf &B)
{
  float theta = state(2);
  float theta_dot = state(3);
  float force = control(0);

  A(0,1) = 1.0;
  A(1,2) = (pole_mass_*cosf(theta)*(pole_length_*powf(theta_dot,2.0)+gravity_*cosf(theta))-gravity_*pole_mass_*powf(sin(theta),2.0))/(cart_mass_+pole_mass_*powf(sinf(theta),2.0))
  -(2*pole_mass_*cosf(theta)*sinf(theta)*(force+pole_mass_*sinf(theta)*(pole_length_*powf(theta_dot,2.0) + gravity_*cosf(theta))))/powf((cart_mass_+pole_mass_*powf(sinf(theta),2.0)),2.0);
  A(1,3) = (2*pole_length_*pole_mass_*theta_dot*sinf(theta))/(cart_mass_+pole_mass_*powf(sinf(theta),2.0));
  A(2,3) = 1.0;
  A(3,2) = (force*sinf(theta)-gravity_*cosf(theta)*(pole_mass_+cart_mass_)-pole_length_*pole_mass_*powf(theta_dot,2.0)*powf(cosf(theta),2.0)+pole_length_*pole_mass_*powf(theta_dot,2.0)*powf(sinf(theta),2.0))/(pole_length_*(cart_mass_+pole_mass_*powf(sinf(theta),2.0)))
  +(2*pole_mass_*cosf(theta)*sinf(theta)*(pole_length_*pole_mass_*cosf(theta)*sinf(theta)*powf(theta_dot,2.0)+force*cosf(theta)+gravity_*sinf(theta)*(pole_mass_+cart_mass_)))/powf(pole_length_*(cart_mass_+pole_mass_*powf(sinf(theta),2.0)),2.0);
  A(3,3) = -(2*pole_mass_*theta_dot*cosf(theta)*sinf(theta))/(cart_mass_+pole_mass_*powf(sinf(theta),2.0));

  B(1,0) = 1/(cart_mass_+pole_mass_*powf(theta,2.0));
  B(3,0) = -cosf(theta)/(pole_length_*(cart_mass_+pole_mass_*powf(sinf(theta),2.0)));
}

void Cartpole::loadParams() {}

void Cartpole::paramsToDevice()
{
  cudaMemcpy(pole_mass_d_, &pole_mass_, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cart_mass_d_, &cart_mass_, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pole_length_d_, &pole_length_, sizeof(float), cudaMemcpyHostToDevice);
}

void Cartpole::freeCudaMem()
{
  cudaFree(pole_mass_d_);
  cudaFree(cart_mass_d_);
  cudaFree(pole_length_d_);
}

void Cartpole::printState(Eigen::MatrixXf state)
{
  printf("Cart position: %f; Cart velocity: %f; Pole angle: %f; Pole rate: %f \n", state(0), state(1), state(2), state(3)); //Needs to be completed
}

void Cartpole::printParams()
{
  printf("Cart mass: %f; Pole mass: %f; Pole length: %f \n", cart_mass_, pole_mass_, pole_length_);
}

__device__ void Cartpole::xDot(float* state, float* control, float* state_der)
{
  float gravity = 9.81;
  float theta = state[2];
  float theta_dot = state[3];
  float force = control[0];
  float m_l = -1;
  float m_c = cart_mass_d_[0];
  float m_p = pole_mass_d_[0];
  float l_p = pole_length_d_[0];


  // TODO WAT?
  state_der[0] = state[1];
  state_der[1] = 1/(m_c+m_l*powf(sinf(theta),2.0))*(force+m_p*sinf(theta)*(l_p*powf(theta_dot,2.0)+gravity*cosf(theta)));
  state_der[2] = state[3];
  state_der[3] = 1/(l_p*(m_c+m_p*powf(sinf(theta),2.0)))*(-force*cosf(theta)-m_p*l_p*powf(theta_dot,2.0)*cosf(theta)*sinf(theta)-(m_c+m_p)*gravity*sinf(theta));
}
