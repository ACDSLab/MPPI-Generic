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
#include <vector>

namespace MPPI_internal {
template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
class Dynamics : public Managed
{
public:
  static const int STATE_DIM = S_DIM;
  static const int CONTROL_DIM = C_DIM;
  static const int SHARED_MEM_REQUEST_GRD = 1; //TODO set to one to prevent array of size 0 error
  static const int SHARED_MEM_REQUEST_BLK = 0;
  float dt_;

  /**
   * useful typedefs
   * TODO: figure out how to ensure matrices passed in to various functions
   * match these types. These values can't be passed in to methods with MatrixXf
   * parameters
   * and MatrixXfs can't be passed into methods with these types because
   * Eigen::Matrix<float, 4, 4> =/= Eigen::MatrixXf(4, 4)!
   */
  typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array; // Control at a time t
  typedef Eigen::Matrix<float, STATE_DIM, 1> state_array; // State at a time t
  typedef Eigen::Matrix<float, STATE_DIM, STATE_DIM> dfdx; // Jacobian wrt x
  typedef Eigen::Matrix<float, STATE_DIM, CONTROL_DIM> dfdu; // Jacobian wrt u
  typedef Eigen::Matrix<float, STATE_DIM, STATE_DIM + CONTROL_DIM> Jacobian; // Jacobian of x and u

  Dynamics() = default;
  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a Dynamics reference
   */
  ~Dynamics() = default;

  /**
   * Allocates all of the GPU memory
   */
  void GPUSetup() {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    if (!GPUMemStatus_) {
      this->model_d_ = Managed::GPUSetup(derived);
    } else {
      std::cout << "GPU Memory already set" << std::endl; //TODO should this be an exception?
    }
    derived->paramsToDevice();
  }

  std::array<float2, C_DIM> getControlRanges() {
    std::array<float2, C_DIM> result;
    for(int i = 0; i < C_DIM; i++) {
      result[i] = control_rngs_[i];
    }
    return result;
  }
  __host__ __device__ float* getControlRangesRaw() {
    return control_rngs_;
  }

  void setParams(const PARAMS_T& params) {
    this->params_ = params;
    if(this->GPUMemStatus_) {
      CLASS_T& derived = static_cast<CLASS_T&>(*this);
      derived.paramsToDevice();
    }
  }

  PARAMS_T getParams() { return params_; }


  /**
   *
   */
  void freeCudaMem() {
    if(GPUMemStatus_) {
      cudaFree(model_d_);
      model_d_ = nullptr;
    }
  }

  /**
   *
   */
  void printState(float* state);

  /**
   *
   */
  void printParams();

  /**
   *
   */
  void paramsToDevice() {
    printf("ERROR: calling paramsToDevice of base dynamics");
    exit(1);
  }

  /**
   * loads the .npz at given path
   * @param model_path
   */
  void loadParams(const std::string& model_path);

  /**
   * updates the internals of the dynamics model
   * @param description
   * @param data
   */
  // TODO generalize
  void updateModel(std::vector<int> description, std::vector<float> data);

  /**
   * compute the Jacobians with respect to state and control
   *
   * @param state   input of current state, passed by reference
   * @param control input of currrent control, passed by reference
   * @param A       output Jacobian wrt state, passed by reference
   * @param B       output Jacobian wrt control, passed by reference
   */
  void computeGrad(const Eigen::MatrixXf& state,
                   const Eigen::MatrixXf& control,
                   Eigen::MatrixXf& A,
                   Eigen::MatrixXf& B);

  /**
   * enforces control constraints
   * @param state
   * @param control
   */
  void enforceConstraints(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  /**
   * updates the current state using s_der
   * @param s state
   * @param s_der
   */
  void updateState(Eigen::MatrixXf &state, Eigen::MatrixXf &s_der, float dt) {
    for (int i = 0; i < STATE_DIM; i++) {
      state(i) += s_der(i)*dt;
      s_der(i) = 0;
    }
  }

  /**
   * computes the section of the state derivative that comes form the dyanmics
   * @param state
   * @param control
   * @param state_der
   */
  void computeDynamics(Eigen::MatrixXf& state, Eigen::MatrixXf& control, Eigen::MatrixXf& state_der);

  /**
   * computes the parts of the state that are based off of kinematics
   * @param s state
   * @param s_der
   */
  void computeKinematics(Eigen::MatrixXf &state, Eigen::MatrixXf &s_der);

  /**
   * computes the full state derivative by calling computeKinematics then computeDynamics
   * @param state
   * @param control
   * @param state_der
   */
  void computeStateDeriv(Eigen::MatrixXf& state, Eigen::MatrixXf& control, Eigen::MatrixXf& state_der);


  /**
   * computes the section of the state derivative that comes form the dyanmics
   * @param state
   * @param control
   * @param state_der
   * @param theta_s shared memory that can be used when computation is computed across the same block
   */
  __device__ void computeDynamics(float* state,
                                float* control,
                                float* state_der,
                                float* theta_s = nullptr);

  /**
   * computes the parts of the state that are based off of kinematics
   * parallelized on X only
   * @param state
   * @param state_der
   */
  __device__ void computeKinematics(float* state, float* state_der) {};

  /**
   * TODO Figure out why this doesn't actually work. Needs to be redefined in
   * derived class
   * computes the full state derivative by calling computeKinematics then computeDynamics
   * @param state
   * @param control
   * @param state_der
   * @param theta_s shared memory that can be used when computation is computed across the same block
   */
  __device__ inline void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s) {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    // only propagate a single state, i.e. thread.y = 0
    // find the change in x,y,theta based off of the rest of the state
    if (threadIdx.y == 0){
      derived->computeKinematics(state, state_der);
    }
    derived->computeDynamics(state, control, state_der, theta_s);
  }

  /**
   * applies the state derivative
   * @param state
   * @param state_der
   * @param dt
   */
  __device__ void updateState(float* state, float* state_der, float dt) {
    for (int i = 0; i < STATE_DIM; i++) {
      state[i] += state_der[i]*dt;
      state_der[i] = 0;
    }
  }

  /**
   * enforces control constraints
   */
  __device__ void enforceConstraints(float* state, float* control);


  // control ranges [.x, .y]
  float2 control_rngs_[C_DIM];

  // device pointer, null on the device
  CLASS_T* model_d_ = nullptr;

  // Eigen matrix holding the state and control jacobians required for DDP N X (N+M)
  // Jacobian jac_;
protected:
  // generic parameter structure
  PARAMS_T params_;
};

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::STATE_DIM;

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::CONTROL_DIM;
} // MPPI_internal
#endif // DYNAMICS_CUH_
