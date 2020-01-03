#pragma once

#ifndef CARTPOLE_CUH_
#define CARTPOLE_CUH_

#include "dynamics.cuh"

struct CartpoleParams {
    float cart_mass = 1.0f;
    float pole_mass = 1.0f;
    float pole_length = 1.0f;

    CartpoleParams();
    CartpoleParams(float cart_mass, float pole_mass, float pole_length):
    cart_mass(cart_mass), pole_mass(pole_mass), pole_length(pole_length) {};
};

class Cartpole : public Dynamics<4, 1>
{
public:
    Cartpole(float delta_t, float cart_mass, float pole_mass, float pole_length, cudaStream_t stream=0);
    ~Cartpole();
    void xDot(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &state_der);  //passing values in by reference
    void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &A, Eigen::MatrixXf &B); //compute the Jacobians with respect to state and control

    void setParams(const CartpoleParams &parameters);
    CartpoleParams getParams();

    __device__ float getCartMass_d();
    __device__ float getPoleMass();
    __device__ float getPoleLength();

    void printState(Eigen::MatrixXf state);
    void printParams();

    __device__ void xDot(float* state, float* control, float* state_der);

    //CUDA parameters
    float* cart_mass_d_;
    float* pole_mass_d_;
    float* pole_length_d_;
    float* gravity_d_;

protected:
    float cart_mass_;
    float pole_mass_;
    float pole_length_;
    const float gravity_ = 9.81;


private:
    void paramsToDevice(); // Params to device should automatically be called when setting parameters
    void freeCudaMem();


};

#endif // CARTPOLE_CUH_
