/*
 * Created on Sun Sep 28 2020 by Bogdan
 */

#ifndef FEEDBACK_CONTROLLERS_DDP_CUH_
#define FEEDBACK_CONTROLLERS_DDP_CUH_

#include <mppi/feedback_controllers/feedback.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <mppi/ddp/ddp_tracking_costs.h>
#include <mppi/ddp/ddp.h>
#include <mppi/ddp/util.h>

template<class DYN_T>
struct DDPParams {
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

  StateCostWeight Q;
  Hessian Q_f;
  ControlCostWeight R;
  int num_iterations = 1;
};

template<class DYN_T>
struct DDPFBState {
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;

  /**
   * Variables
   **/
  feedback_gain_trajectory fb_gain_traj_ = feedback_gain_trajectory(0);
};

template<class DYN_T, int N_TIMESTEPS>
struct DDPFeedbackState {
  static const int FEEDBACK_SIZE = DYN_T::CONTROL_DIM * DYN_T::STATE_DIM * N_TIMESTEPS;
  static const int NUM_TIMESTEPS = N_TIMESTEPS;

  /**
   * Variables
   **/
  float fb_gain_traj_[FEEDBACK_SIZE] = {0.0};
};

/**
  * DDP GPU Controller class starting point. This class is where the actual
  * methods for DDP on the GPU are implemented but it is not used directly since
  * setting up the GPU_FB_T value would be painful
  */
template <class GPU_FB_T, class DYN_T, int NUM_TIMESTEPS = 1>
class DeviceDDPImpl : public GPUFeedbackController<
  DeviceDDPImpl<GPU_FB_T, DYN_T, NUM_TIMESTEPS>, DYN_T, DDPFeedbackState<DYN_T, NUM_TIMESTEPS>> {
public:

  // using TEMPLATED_DYNAMICS = typename GPUFeedbackController<DeviceDDPImpl<GPU_FB_T, DYN_T>, DYN_T>::DYN_T;

  float * fb_gains_ = nullptr;
  float * fb_gains_d_ = nullptr;
  DeviceDDPImpl(int num_timesteps, cudaStream_t stream = 0);
  DeviceDDPImpl(cudaStream_t stream = 0) : GPUFeedbackController<DeviceDDPImpl<GPU_FB_T, DYN_T, NUM_TIMESTEPS>, DYN_T, DDPFeedbackState<DYN_T, NUM_TIMESTEPS>>(stream) {};

  void allocateCUDAMemory() {};
  void deallocateCUDAMemory() {};

  __device__ void k(const float * x_act, const float * x_goal,
                           const float t, float * theta,
                           float* control_output);
  // void copyToDevice();
  // Nothing to copy back
  void copyFromDevice() {}
protected:
  // Needed for allocating memory for feedback gains
  int num_timesteps_ = 1;
};

/**
  * Alias class for DDP GPU Controller. This sets up the class derivation correctly and is
  * used inside of the CPU version of DDP
  */
template <class DYN_T, int NUM_TIMESTEPS>
class DeviceDDP : public DeviceDDPImpl<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DYN_T,
                                       NUM_TIMESTEPS> {
public:
  DeviceDDP(int num_timesteps, cudaStream_t stream=0) :
    DeviceDDPImpl<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DYN_T, NUM_TIMESTEPS>(num_timesteps, stream) {};

  DeviceDDP(cudaStream_t stream=0) :
    DeviceDDPImpl<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DYN_T, NUM_TIMESTEPS>(stream) {};
};


/**
  * CPU Class for DDP. This is what the user should interact with
  */
template <class DYN_T, int NUM_TIMESTEPS>
class DDPFeedback : public FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>,
                                              DDPParams<DYN_T>,
                                              NUM_TIMESTEPS> {
public:
  /**
   * Aliases
   **/
  // typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;
  using feedback_gain_trajectory = typename DDPFBState<DYN_T>::feedback_gain_trajectory;

  using control_array = typename FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>,
                                                    DDPParams<DYN_T>,
                                                    NUM_TIMESTEPS>::control_array;
  using state_array = typename FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>,
                                                  DDPParams<DYN_T>,
                                                  NUM_TIMESTEPS>::state_array;
  using state_trajectory = typename FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>,
                                                       DDPParams<DYN_T>,
                                                       NUM_TIMESTEPS>::state_trajectory;
  using control_trajectory = typename FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DDPParams<DYN_T>,
                                                         NUM_TIMESTEPS>::control_trajectory;
  using INTERNAL_STATE_T = typename FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DDPParams<DYN_T>,
                                                       NUM_TIMESTEPS>::TEMPLATED_FEEDBACK_STATE;
  using feedback_gain_matrix = typename DYN_T::feedback_matrix;
  using square_state_matrix = typename DDPParams<DYN_T>::StateCostWeight;
  using square_control_matrix = typename DDPParams<DYN_T>::ControlCostWeight;

  /**
   * Variables
   **/
  // feedback_gain_trajectory fb_gain_traj_;
  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;
  OptimizerResult<ModelWrapperDDP<DYN_T>> result_;

  control_array control_min_;
  control_array control_max_;
  DYN_T* model_;


  DDPFeedback(DYN_T* model, float dt, int num_timesteps = NUM_TIMESTEPS,
              cudaStream_t stream = 0);

  /**
   * Copy operator for DDP controller
   */
  // DDPFeedback<DYN_T, NUM_TIMESTEPS>& operator=(const DDPFeedback<DYN_T, NUM_TIMESTEPS>& other) {
  //   if (this != other) {
  //     // if (ddp_model_ != 0) {
  //     //   *ddp_model_ = *other.ddp_model_;
  //     // } else {
  //     //   ddp_model_ = std::make_shared<ModelWrapperDDP<DYN_T>>(other.model_);
  //     //   *ddp_model_ = *other.ddp_model_;
  //     // }

  //     // if (run_cost_ != 0) {
  //     //   *run_cost_ = *other.run_cost_;
  //     // } else {
  //     //   run_cost_ = std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(this->params_.Q,
  //     //                                                                         this->params_.R,
  //     //                                                                         NUM_TIMESTEPS);
  //     //   *run_cost_ = *other.run_cost_;
  //     // }

  //     bool tracking_uninitialized = ddp_model_ == nullptr ||
  //                                   run_cost_ == nullptr ||
  //                                   terminal_cost_ == nullptr ||
  //                                   ddp_solver_ == nullptr;
  //     // No memory has been allocated yet
  //     if (tracking_uninitialized) {
  //       initTrackingController();
  //     }
  //     // Deep copy of pointer variables
  //     *ddp_model_ = *other.ddp_model_;
  //     *run_cost_ = *other.run_cost_;
  //     *terminal_cost_ = *other.terminal_cost_;
  //     *ddp_solver_ = *other.ddp_solver_;

  //     // TODO Figure out what to do about GPU portion
  //     gpu_controller_ = nullptr;

  //     // Copy of remaining variables
  //     result_ = other.result_;
  //     fb_gain_traj_ = other.fb_gain_traj_;
  //     control_min_ = other.control_min_;
  //     control_max_ = other.control_max_;
  //   }
  // }

  void setParams(DDPParams<DYN_T>& params);

  void initTrackingController();

  control_array k_(const Eigen::Ref<const state_array>& x_act,
                   const Eigen::Ref<const state_array>& x_goal, float t,
                   INTERNAL_STATE_T& fb_state) {
    int index = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * int(t);
    Eigen::Map<feedback_gain_matrix> fb_gain(&(fb_state.fb_gain_traj_[index]));
    control_array u_output = fb_gain * (x_act - x_goal);
    return u_output;
  }

  void computeFeedbackGains(const Eigen::Ref<const state_array>& init_state,
                            const Eigen::Ref<const state_trajectory>& goal_traj,
                            const Eigen::Ref<const control_trajectory>& control_traj);
};

#ifdef __CUDACC__
#include "ddp.cu"
#endif

#endif  // FEEDBACK_CONTROLLERS_DDP_CUH_
