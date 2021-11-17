#pragma once

#ifndef DYNAMICS_CUH_
#define DYNAMICS_CUH_

/*
Header file for dynamics
*/

#include <Eigen/Dense>
#include <stdio.h>
#include <math.h>
#include <mppi/utils/managed.cuh>
#include <vector>
#include <cfloat>

namespace MPPI_internal {
template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
class Dynamics : public Managed
{
public:
//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int STATE_DIM = S_DIM;
  static const int CONTROL_DIM = C_DIM;
  static const int SHARED_MEM_REQUEST_GRD = 1; //TODO set to one to prevent array of size 0 error
  static const int SHARED_MEM_REQUEST_BLK = 0;
  typedef CLASS_T DYN_T;
  typedef PARAMS_T DYN_PARAMS_T;

  /**
   * useful typedefs
   */
  typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array; // Control at a time t
  typedef Eigen::Matrix<float, STATE_DIM, 1> state_array; // State at a time t
  typedef Eigen::Matrix<float, STATE_DIM, STATE_DIM> dfdx; // Jacobian wrt x
  typedef Eigen::Matrix<float, STATE_DIM, CONTROL_DIM> dfdu; // Jacobian wrt u
  typedef Eigen::Matrix<float, CONTROL_DIM, STATE_DIM> feedback_matrix; // Feedback matrix
  typedef Eigen::Matrix<float, STATE_DIM, STATE_DIM + CONTROL_DIM> Jacobian; // Jacobian of x and u

  // protected constructor prevent anyone from trying to construct a Dynamics
protected:
  /**
   * sets the default control ranges to -infinity and +infinity
   */
  Dynamics(cudaStream_t stream=0) : Managed(stream) {
    // TODO handle at Managed
    for(int i = 0; i < C_DIM; i++) {
      control_rngs_[i].x = -FLT_MAX;
      control_rngs_[i].y = FLT_MAX;
    }
  }

  /**
   * sets the control ranges to the passed in value
   * @param control_rngs
   * @param stream
   */
  Dynamics(std::array<float2, C_DIM>& control_rngs, cudaStream_t stream=0) : Managed(stream) {
    setControlRanges(control_rngs);
  }

  Dynamics(PARAMS_T& params, std::array<float2, C_DIM>& control_rngs, cudaStream_t stream=0) : Managed(stream) {
    setParams(params);
    setControlRanges(control_rngs);
  }

  Dynamics(PARAMS_T& params, cudaStream_t stream=0) : Managed(stream) {
    setParams(params);
  }
public:
  // This variable defines what the zero control is
  // For most systems, it should be zero but for things like a quadrotor,
  // it should be the command to hover
  control_array zero_control_ = control_array::Zero();

  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a Dynamics reference
   */
  virtual ~Dynamics() {
    freeCudaMem();
  }

  /**
   * Allocates all of the GPU memory
   */
  void GPUSetup();

  std::array<float2, C_DIM> getControlRanges() {
    std::array<float2, C_DIM> result;
    for(int i = 0; i < C_DIM; i++) {
      result[i] = control_rngs_[i];
    }
    return result;
  }
  __host__ __device__ float2* getControlRangesRaw() {
    return control_rngs_;
  }

  void setControlRanges(std::array<float2, C_DIM>& control_rngs) {
    for(int i = 0; i < C_DIM; i++) {
      control_rngs_[i].x = control_rngs[i].x;
      control_rngs_[i].y = control_rngs[i].y;
    }
    if(GPUMemStatus_) {
      HANDLE_ERROR( cudaMemcpy(this->model_d_->control_rngs_, this->control_rngs_, C_DIM*sizeof(float2), cudaMemcpyHostToDevice) );
    }
  }

  void setParams(const PARAMS_T& params) {
    params_ = params;
    if(GPUMemStatus_) {
      CLASS_T& derived = static_cast<CLASS_T&>(*this);
      derived.paramsToDevice();
    }
  }

  __device__ __host__ PARAMS_T getParams() { return params_; }


  /*
   *
   */
  void freeCudaMem();

  /**
   *
   */
  void printState(float* state);

  /**
   *
   */
   // TODO should not assume it is going to cout, pass in stream
  void printParams();

  /**
   *
   */
  void paramsToDevice();

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
  bool computeGrad(const Eigen::Ref<const state_array>& state = state_array(),
                   const Eigen::Ref<const control_array>& control = control_array(),
                   Eigen::Ref<dfdx> A = dfdx(),
                   Eigen::Ref<dfdu> B = dfdu()) {return false;}
  /**
   * enforces control constraints
   * @param state
   * @param control
   */
  void enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control) {
    for(int i = 0; i < C_DIM; i++) {
      //printf("enforceConstraints %f, min = %f, max = %f\n", control(i), control_rngs_[i].x, control_rngs_[i].y);
      if(control(i) < control_rngs_[i].x) {
        control(i) = control_rngs_[i].x;
      } else if(control(i) > control_rngs_[i].y) {
        control(i) = control_rngs_[i].y;
      }
    }
  }

  /**
   * updates the current state using s_der
   * @param s state
   * @param s_der
   */
  void updateState(Eigen::Ref<state_array> state,
                   Eigen::Ref<state_array> state_der, const float dt) {
    state += state_der * dt;
    state_der.setZero();
  }

  /**
   * does a linear interpolation of states
   * @param state_1
   * @param state_2
   * @param alpha
   * @return
   */
  state_array interpolateState(const Eigen::Ref<state_array> state_1,
                                           const Eigen::Ref<state_array> state_2, const double alpha) {
    return (1 - alpha)*state_1 + alpha*state_2;
  }

  /**
   * computes the section of the state derivative that comes form the dyanmics
   * @param state
   * @param control
   * @param state_der
   */
  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der);

  /**
   * computes the parts of the state that are based off of kinematics
   * @param s state
   * @param s_der
   */
  void computeKinematics(const Eigen::Ref<const state_array>& state, Eigen::Ref<state_array> s_der) {};

  /**
   * computes the full state derivative by calling computeKinematics then computeDynamics
   * @param state
   * @param control
   * @param state_der
   */
  void computeStateDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der) {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->computeKinematics(state, state_der);
    derived->computeDynamics(state, control, state_der);
  }

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
      //printf("state at 0 before kin: %f\n", state[0]);
      derived->computeKinematics(state, state_der);
      //printf("state at 0 after kin: %f\n", state[0]);
    }
    derived->computeDynamics(state, control, state_der, theta_s);
    //printf("state at 0 after dyn: %f\n", state[0]);
  }

  /**
   * applies the state derivative
   * @param state
   * @param state_der
   * @param dt
   */
  __device__ void updateState(float* state, float* state_der, const float dt) {
    int i;
    int tdy = threadIdx.y;
    //Add the state derivative time dt to the current state.
    //printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
    for (i = tdy; i < STATE_DIM; i+=blockDim.y){
      state[i] += state_der[i]*dt;
      state_der[i] = 0; //Important: reset the state derivative to zero.
    }
  }

  /**
   * enforces control constraints
   */
  __device__ void enforceConstraints(float* state, float* control) {
    // TODO should control_rngs_ be a constant memory parameter
    int i;
    int tdy = threadIdx.y;
    // parallelize setting the constraints with y dim
    for (i = tdy; i < CONTROL_DIM; i+=blockDim.y){
      //printf("thread index = %d, %d, control %f\n", threadIdx.x, tdy, control[i]);
      if(control[i] < control_rngs_[i].x) {
        control[i] = control_rngs_[i].x;
      } else if(control[i] > control_rngs_[i].y) {
        control[i] = control_rngs_[i].y;
      }
      //printf("finished thread index = %d, %d, control %f\n", threadIdx.x, tdy, control[i]);
    }
  }


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

#ifdef __CUDACC__
#include "dynamics.cu"
#endif

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::STATE_DIM;

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::CONTROL_DIM;

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::SHARED_MEM_REQUEST_BLK;

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
const int Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::SHARED_MEM_REQUEST_GRD;
} // MPPI_internal
#endif // DYNAMICS_CUH_
