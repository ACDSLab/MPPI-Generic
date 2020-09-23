/*
 * Created on Sun Sep 6 2020 by Bogdan
 */

#ifndef FEEDBACK_BASE_CUH_
#define FEEDBACK_BASE_CUH_

#include <Eigen/Core>
#include <mppi/utils/managed.cuh>

#include <cmath>
#include <memory>
#include <tuple>

template<class CLASS_T, class DYN_T> class GPUFeedbackController : public Managed {
public:

  virtual ~GPUFeedbackController() {
    deallocateCUDAMemory();
  };

  // Overwrite of Managed->GPUSetup to call allocateCUDAMemory as well
  void GPUSetup();
  // Method to allocate more CUDA memory if needed
  void allocateCUDAMemory();
  void deallocateCUDAMemory();

  void k(const float * x_act, const float * x_goal,
         const float t,  const float * theta, float* control_output);

  // Abstract method to copy information to GPU
  void copyToDevice();
  void copyFromDevice();
  // Needs the CLASS_T pointer to itself
}

/**
 * Steps to making a new one
 * Create the GPUFeedback class as an impl class like costs but is still templated on DYN
 * The actual GPUFeedback_act class will then be templated on DYN and inherit from the GPUFeedbackImpl
 * Write the feedback controller to use the GPUFeedback_act as thee GPU_FEEDBACK_T template option
 * It will then automatically create the right pointer
 */

template<class DYN_T, template<class> GPU_FEEDBACK_T, int NUM_TIMESTEPS> class FeedbackController {
public:
  // Type Defintions and aliases

  using state_array = typename DYN_T::state_array;
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, NUM_TIMESTEPS> control_trajectory; // A control trajectory
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS> state_trajectory; // A control trajectory
  typedef DYN_T TEMPLATED_DYNAMICS;
  typedef GPU_FEEDBACK_T<DYN_T> GPUFeedback;

  // Constructors and Generators
  FeedbackController(cudaStream_t stream=0) {
    gpu_controller = std::make_shared<GPUFeedback>();
  }

  virtual ~FeedbackController() = default;



  // CPU Methods
  virtual control_array k(const Eigen::Ref<state_array>& x_act,
                          const Eigen::Ref<state_array>& x_goal, float t) = 0;

  // might not be a needed method
  virtual computeFeedbackGains(const Eigen::Ref<const state_array>& init_state,
                               const Eigen::Ref<const state_trajector>y& goal_traj,
                               const Eigen::Ref<const control_trajectory>& control_traj);
  // Abstract method to copy information to GPU using the copyToDevice method in gpu_controller
  virtual sendToGPU() = 0;
  virtual copyFromGPU() = 0;
  GPUFeeback* getDevicePointer() {
    return gpu_controller->
  }
private:
  std::shared_ptr<GPUFeedback> gpu_controller;
}


#endif // FEEDBACK_BASE_CUH_