#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>

CartpoleDynamics::CartpoleDynamics(float cart_mass, float pole_mass, float pole_length, cudaStream_t stream) : Dynamics<CartpoleDynamics, CartpoleDynamicsParams, 4, 1>(stream)
{
    this->params_ = CartpoleDynamicsParams(cart_mass, pole_mass, pole_length);
}

CartpoleDynamics::~CartpoleDynamics() {
}


void CartpoleDynamics::computeGrad(const Eigen::Ref<const state_array> & state,
                                   const Eigen::Ref<const control_array>& control,
                                   Eigen::Ref<dfdx> A,
                                   Eigen::Ref<dfdu> B) {
  float theta = state(2);
  float theta_dot = state(3);
  float force = control(0);

  A(0,1) = 1.0;
  A(1,2) = (this->params_.pole_mass*cosf(theta)*(this->params_.pole_length*powf(theta_dot,2.0)+gravity_*cosf(theta))-gravity_*this->params_.pole_mass*powf(sin(theta),2.0))/(this->params_.cart_mass+this->params_.pole_mass*powf(sinf(theta),2.0))
  -(2*this->params_.pole_mass*cosf(theta)*sinf(theta)*(force+this->params_.pole_mass*sinf(theta)*(this->params_.pole_length*powf(theta_dot,2.0) + gravity_*cosf(theta))))/powf((this->params_.cart_mass+this->params_.pole_mass*powf(sinf(theta),2.0)),2.0);
  A(1,3) = (2*this->params_.pole_length*this->params_.pole_mass*theta_dot*sinf(theta))/(this->params_.cart_mass+this->params_.pole_mass*powf(sinf(theta),2.0));
  A(2,3) = 1.0;
  A(3,2) = (force*sinf(theta)-gravity_*cosf(theta)*(this->params_.pole_mass+this->params_.cart_mass)-this->params_.pole_length*this->params_.pole_mass*powf(theta_dot,2.0)*powf(cosf(theta),2.0)+this->params_.pole_length*this->params_.pole_mass*powf(theta_dot,2.0)*powf(sinf(theta),2.0))/(this->params_.pole_length*(this->params_.cart_mass+this->params_.pole_mass*powf(sinf(theta),2.0)))
  +(2*this->params_.pole_mass*cosf(theta)*sinf(theta)*(this->params_.pole_length*this->params_.pole_mass*cosf(theta)*sinf(theta)*powf(theta_dot,2.0)+force*cosf(theta)+gravity_*sinf(theta)*(this->params_.pole_mass+this->params_.cart_mass)))/powf(this->params_.pole_length*(this->params_.cart_mass+this->params_.pole_mass*powf(sinf(theta),2.0)),2.0);
  A(3,3) = -(2*this->params_.pole_mass*theta_dot*cosf(theta)*sinf(theta))/(this->params_.cart_mass+this->params_.pole_mass*powf(sinf(theta),2.0));

  B(1,0) = 1/(this->params_.cart_mass+this->params_.pole_mass*powf(theta,2.0));
  B(3,0) = -cosf(theta)/(this->params_.pole_length*(this->params_.cart_mass+this->params_.pole_mass*powf(sinf(theta),2.0)));
}

void CartpoleDynamics::computeDynamics(const Eigen::Ref<const state_array> &state,
                                       const Eigen::Ref<const control_array> &control,
                                       Eigen::Ref<state_array> state_der) {
  float theta = state(2);
  float theta_dot = state(3);
  float force = control(0);
  float m_c = this->params_.cart_mass;
  float m_p = this->params_.pole_mass;
  float l_p = this->params_.pole_length;

  // TODO WAT?
  state_der(0) = state(1);
  state_der(1) = 1/(m_c+m_p*powf(sinf(theta),2.0))*(force+m_p*sinf(theta)*(l_p*powf(theta_dot,2.0)+gravity_*cosf(theta)));
  state_der(2) = state(3);
  state_der(3) = 1/(l_p*(m_c+m_p*powf(sinf(theta),2.0)))*(-force*cosf(theta)-m_p*l_p*powf(theta_dot,2.0)*cosf(theta)*sinf(theta)-(m_c+m_p)*gravity_*sinf(theta));
}

void CartpoleDynamics::freeCudaMem()
{
  Dynamics<CartpoleDynamics, CartpoleDynamicsParams, 4, 1>::freeCudaMem();
}

void CartpoleDynamics::printState(const Eigen::Ref<const state_array>& state)
{
  printf("Cart position: %f; Cart velocity: %f; Pole angle: %f; Pole rate: %f \n", state(0), state(1), state(2), state(3)); //Needs to be completed
}

void CartpoleDynamics::printState(float *state) {
    printf("Cart position: %f; Cart velocity: %f; Pole angle: %f; Pole rate: %f \n", state[0], state[1], state[2], state[3]); //Needs to be completed
}

void CartpoleDynamics::printParams()
{
  printf("Cart mass: %f; Pole mass: %f; Pole length: %f \n", this->params_.cart_mass, this->params_.pole_mass, this->params_.pole_length);
}

__device__ void CartpoleDynamics::computeDynamics(float* state, float* control,
                                                  float* state_der, float* theta_s)
{
  float theta = state[2];
  float theta_dot = state[3];
  float force = control[0];
  float m_c = this->params_.cart_mass;
  float m_p = this->params_.pole_mass;
  float l_p = this->params_.pole_length;

  state_der[0] = state[1];
  state_der[1] = 1/(m_c+m_p*powf(sinf(theta),2.0))*(force+m_p*sinf(theta)*(l_p*powf(theta_dot,2.0)+gravity_*cosf(theta)));
  state_der[2] = state[3];
  state_der[3] = 1/(l_p*(m_c+m_p*powf(sinf(theta),2.0)))*(-force*cosf(theta)-m_p*l_p*powf(theta_dot,2.0)*cosf(theta)*sinf(theta)-(m_c+m_p)*gravity_*sinf(theta));
}

//void CartpoleDynamics::computeStateDeriv(const state_array &state, const control_array &control, state_array &state_der) {
//  computeDynamics(state, control, state_der);
//}
