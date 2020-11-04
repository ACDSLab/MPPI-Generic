/*
 * Created on Sun Sep 28 2020 by Bogdan
 */

#ifndef FEEDBACK_CONTROLLERS_DDP_CUH_
#define FEEDBACK_CONTROLLERS_DDP_CUH_

#include <mppi/feedback_controllers/feedback.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <mppi/ddp/ddp_tracking_costs.h>
#include <mppi/ddp/ddp.h>

template<class DYN_T>
struct DDPParams {
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

  StateCostWeight Q;
  Hessian Q_f;
  ControlCostWeight R;
};

template<class DYN_T>
struct DDPFBState {
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;

  /**
   * Variables
   **/
  feedback_gain_trajectory fb_gain_traj_ = feedback_gain_trajectory::Zero();
};

// Class where methods are implemented
template <class GPU_FB_T, class DYN_T>
class DeviceDDPImpl : public GPUFeedbackController<DeviceDDPImpl<GPU_FB_T, DYN_T>, DYN_T> {
public:
  DeviceDDPImpl(int num_timesteps, cudaStream_t stream = 0);

  void allocateCUDAMemory();
  void deallocateCUDAMemory();

  void k(const float * x_act, const float * x_goal,
         const float t, float * theta,
         float* control_output);
  void copyToDevice();
  // Nothing to copy back
  void copyFromDevice() {}
protected:
  float * fb_gains_ = nullptr;
  // Needed for allocating memory for feedback gains
  int num_timesteps_ = 0;
};

// Alias class for
template <class DYN_T>
class DeviceDDP : public DeviceDDPImpl<DeviceDDP<DYN_T>, DYN_T> {
public:
  ///DeviceDDP(cudaStream_t stream=0) : DeviceDDPImpl<DeviceDDP, DYN_T>(stream) {};
};


template <class DYN_T, int NUM_TIMESTEPS>
class DDPFeedback : public FeedbackController<DeviceDDP<DYN_T>, DDPParams<DYN_T>,
                                              DDPFBState<DYN_T>, NUM_TIMESTEPS> {
public:
  /**
   * Aliases
   **/
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;
  using feedback_gain_trajectory = DDPFBState<DYN_T>::feedback_gain_trajectory;

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


  DDPFeedback(cudaStream_t stream = 0);

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

  void initTrackingController();

  control_array k(const Eigen::Ref<state_array>& x_act,
                  const Eigen::Ref<state_array>& x_goal, float t,
                  INTERAL_STATE_T& fb_state);

  void computeFeedbackGains(const Eigen::Ref<const state_array>& init_state,
                            const Eigen::Ref<const state_trajectory>& goal_traj,
                            const Eigen::Ref<const control_trajectory>& control_traj);

  control_array interpolateFeedback(state_array& state, state_array& target_nominal_state,
                                    double rel_time, INTERAL_STATE_T& fb_state);
};

#ifdef __CUDACC__
#include "ddp.cu"
#endif

#endif  // FEEDBACK_CONTROLLERS_DDP_CUH_
