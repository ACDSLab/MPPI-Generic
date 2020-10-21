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
class DDPFeedback : public FeedbackController<DeviceDDP<DYN_T>, DDPParams<DYN_T>, NUM_TIMESTEPS> {
public:
  /**
   * Aliases
   **/
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;

  /**
   * Variables
   **/
  feedback_gain_trajectory fb_gain_traj_;
  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;

  DDPFeedback();

  void initTrackingController();

  control_array k(const Eigen::Ref<state_array>& x_act,
                  const Eigen::Ref<state_array>& x_goal, float t);

  void computeFeedbackGains(const Eigen::Ref<const state_array>& init_state,
                            const Eigen::Ref<const state_trajectory>& goal_traj,
                            const Eigen::Ref<const control_trajectory>& control_traj);

  control_array interpolateFeedback(state_array& state, state_array& target_nominal_state,
                                    feedback_gain_trajectory& gain_traj, double rel_time);
};

#ifdef __CUDACC__
#include "ddp.cu"
#endif

#endif  // FEEDBACK_CONTROLLERS_DDP_CUH_
