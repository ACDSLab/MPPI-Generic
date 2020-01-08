#pragma once

#ifndef CARTPOLE_CUH_
#define CARTPOLE_CUH_

#include <dynamics/dynamics.cuh>

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

    void GPUSetup();

    void xDot(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &state_der);  //passing values in by reference
    void computeGrad(Eigen::MatrixXf &state, Eigen::MatrixXf &control, Eigen::MatrixXf &A, Eigen::MatrixXf &B); //compute the Jacobians with respect to state and control

    void setParams(const CartpoleParams &parameters);
    CartpoleParams getParams();

    __host__ __device__ float getCartMass() {return cart_mass_;};
    __host__ __device__ float getPoleMass() {return pole_mass_;};
    __host__ __device__ float getPoleLength() {return pole_length_;};
    __host__ __device__ float getGravity() {return gravity_;}

    void printState(Eigen::MatrixXf state);
    void printParams();

    __device__ void xDot(float* state, float* control, float* state_der);

    // Device pointer of class object
    Cartpole* CP_device = nullptr;

    void freeCudaMem();



protected:
    float cart_mass_;
    const float gravity_ = 9.81;
    float pole_mass_;
    float pole_length_;

    void paramsToDevice(); // Params to device should automatically be called when setting parameters

};

#if __CUDACC__
#include "cartpole.cu"
#endif

#endif // CARTPOLE_CUH_


