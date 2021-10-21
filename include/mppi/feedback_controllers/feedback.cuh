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

/**
  * This is the begining of the mess. So we have a GPU Feedback class that houses the methods and data
  * that are used on the GPU. The first template argument is for that cool thing that Jason knows the name of.
  * The second template argument is the dynamics class so that we can automatically pull out the state and control
  * dims needed. The final template argument is the state class. This state class is what will be passed back and forth
  * from the CPU to the GPU and vice versa (if necessary). For example, this would be your k_p, k_d, and k_i terms for a PID,
  * or the trajectory of feedback gains for DDP.
  *
  * Things a new controller will need:
  *   - a k method. This is how to use the feedback controller on the GPU
  *   - a GPU_STATE_T class. This should contain the relevant data to transfer from CPU to GPU and vice versa
  * Optional Things to implement:
  *   - copyFromDevice(), copyToDevice(), paramsToDevice(): if your data structure FEEDBACK_STATE_T is complex, you will need to implement these yourself
  *   - allocateCUDAMemory(), deallocateCUDAMemory(): If your feedback class needs some CUDA memory beyond the FEEDBACK_STATE_T, you will need to create and clear it here.
  */
template<class GPU_FB_T, class TEMPLATED_DYNAMICS, class GPU_STATE_T>
class GPUFeedbackController : public Managed {
public:
  /**
   * Type Aliasing
   */
  using DYN_T = TEMPLATED_DYNAMICS;
  using FEEDBACK_STATE_T = GPU_STATE_T;

  GPU_FB_T* feedback_d_ = nullptr;

  /**
   * Constructors
   */

  GPUFeedbackController() = default;

  GPUFeedbackController(cudaStream_t stream = 0) : Managed(stream) {}

  /**
   * =================== METHODS THAT SHOULD NOT BE OVERWRITTEN ================
   */
  virtual ~GPUFeedbackController() {
    freeCudaMem();
  };

  // Overwrite of Managed->GPUSetup to call allocateCUDAMemory as well
  void GPUSetup();
  void freeCudaMem();

  void setFeedbackState(FEEDBACK_STATE_T state) {
    state_ = state;
    if (GPUMemStatus_) {
      GPU_FB_T& derived = static_cast<GPU_FB_T&>(*this);
      derived.copyToDevice();
    }
  }

  __host__ __device__ FEEDBACK_STATE_T getFeedbackState() {
    return state_;
  }

  __host__ __device__ FEEDBACK_STATE_T* getFeedbackStatePointer() {
    return &state_;
  }

  /**
   * ==================== NECESSARY METHODS TO OVERWRITE =====================
   */
  __device__ void k(const float * x_act, const float * x_goal,
         const float t, float * theta,
         float* control_output) {
    //CLASS_T* derived = static_cast<CLASS_T*>(this);
    //derived->k(x_act, x_goal, t, theta, control_output);
  }
  /**
   * ===================== OPTIONAL METHODS TO OVERWRITE ======================
   */
  /**
   * Only needed to allocate/deallocate additional CUDA memory when appropriate, GPU pointer is already handled.
   */
  void allocateCUDAMemory() {}
  void deallocateCUDAMemory() {}

  // Abstract method to copy information to GPU
  // void copyToDevice() {}

  // Copies the params to the device at the moment
  void copyToDevice();
  // Method to return potential diagnostic information from GPU
  void copyFromDevice() {}
protected:
  FEEDBACK_STATE_T state_;
};

/**
 * Steps to making a new one
 * Create the GPUFeedback class as an impl class like costs but is still templated on DYN
 * The actual GPUFeedback_act class will then be templated on DYN and inherit from the GPUFeedbackImpl
 * Write the feedback controller to use the GPUFeedback_act as thee GPU_FEEDBACK_T template option
 * It will then automatically create the right pointer
 */
template<class GPU_FB_T, class PARAMS_T, int NUM_TIMESTEPS>
class FeedbackController {
public:

  // Type Defintions and aliases
  typedef typename GPU_FB_T::DYN_T DYN_T;
  // typedef FEEDBACK_STATE_T TEMPLATED_FEEDBACK_STATE;
  typedef PARAMS_T TEMPLATED_PARAMS;
  typedef GPU_FB_T TEMPLATED_GPU_FEEDBACK;
  typedef typename GPU_FB_T::FEEDBACK_STATE_T TEMPLATED_FEEDBACK_STATE;

  using state_array = typename DYN_T::state_array;
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM,
                        NUM_TIMESTEPS> control_trajectory; // A control trajectory
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM,
                        NUM_TIMESTEPS> state_trajectory; // A state trajectory

  // Constructors and Generators
  FeedbackController(float dt = 0.01, int num_timesteps = NUM_TIMESTEPS,
                     cudaStream_t stream=0) :  dt_(dt), num_timesteps_(num_timesteps) {
    gpu_controller_ = std::make_shared<GPU_FB_T>(stream);
    gpu_controller_->GPUSetup();
  }

  virtual ~FeedbackController() = default;

  virtual void initTrackingController() = 0;

  virtual void setParams(PARAMS_T& params) {
    params_ = params;
  }

  // CPU Methods
  // TODO Construct a default version of this method that uses the state_ variable automatically
  virtual control_array k(const Eigen::Ref<state_array>& x_act,
                          const Eigen::Ref<state_array>& x_goal, float t,
                          TEMPLATED_FEEDBACK_STATE& fb_state) = 0;

  // might not be a needed method
  virtual void computeFeedbackGains(const Eigen::Ref<const state_array>& init_state,
                                    const Eigen::Ref<const state_trajectory>& goal_traj,
                                    const Eigen::Ref<const control_trajectory>& control_traj) = 0;

  // TODO Construct a default version of this method that uses the state_ variable automatically
  virtual control_array interpolateFeedback(state_array& state, state_array& target_nominal_state,
                                            double rel_time, TEMPLATED_FEEDBACK_STATE& fb_state) {
    // TODO call the feedback controller version directly
    int lower_idx = (int) (rel_time / dt_);
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * dt_) / dt_;

    control_array u_fb = (1 - alpha) * k(state, target_nominal_state, lower_idx, fb_state)
        + alpha*k(state, target_nominal_state, upper_idx, fb_state);

    return u_fb;
  }

  GPU_FB_T* getDevicePointer() {
    return gpu_controller_->feedback_d_;
  }

  std::shared_ptr<GPU_FB_T> getHostPointer() {
    return gpu_controller_;
  }

  void bindToStream(cudaStream_t stream) {
    gpu_controller_->bindToStream(stream);
  }

  /**
   * Calls GPU version
   */
  void copyToDevice() {
    this->gpu_controller_->copyToDevice();
  }

  TEMPLATED_FEEDBACK_STATE getFeedbackState() { // TODO change to getFeedbackState()
    return this->gpu_controller_->getFeedbackState();
  }

  TEMPLATED_FEEDBACK_STATE* getFeedbackStatePointer() { // TODO change to getFeedbackState()
    return this->gpu_controller_->getFeedbackStatePointer();
  }

  void setFeedbackState(TEMPLATED_FEEDBACK_STATE& gpu_fb_state) {
    this->gpu_controller_->setFeedbackState(gpu_fb_state);
  }
protected:
  std::shared_ptr<GPU_FB_T> gpu_controller_;
  float dt_;
  int num_timesteps_;
  PARAMS_T params_;
  // FEEDBACK_STATE_T feedback_state_;
};

#ifdef __CUDACC__
#include "feedback.cu"
#endif

#endif // FEEDBACK_BASE_CUH_
