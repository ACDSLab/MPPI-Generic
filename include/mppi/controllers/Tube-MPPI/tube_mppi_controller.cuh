/**
  * Created by Manan Gandhi on 2/14/2020
  * API for interfacing with the TUBE MPPI controller.
  */

#ifndef MPPIGENERIC_TUBE_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_TUBE_MPPI_CONTROLLER_CUH

#include <curand.h>
#include <mppi/controllers/controller.cuh>
#include <mppi/core/mppi_common.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <mppi/ddp/ddp_tracking_costs.h>
#include <mppi/ddp/ddp.h>
#include <chrono>
#include <memory>
#include <iostream>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y>
class TubeMPPIController: public Controller<DYN_T, COST_T,
                                            MAX_TIMESTEPS,
                                            NUM_ROLLOUTS,
                                            BDIM_X,
                                            BDIM_Y> {

public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW unnecessary due to EIGEN_MAX_ALIGN_BYTES=0
  /**
   * Set up useful types
   */
  using control_array = typename Controller<DYN_T, COST_T,
          MAX_TIMESTEPS,
          NUM_ROLLOUTS,
          BDIM_X,
          BDIM_Y>::control_array;

  using control_trajectory = typename Controller<DYN_T, COST_T,
          MAX_TIMESTEPS,
          NUM_ROLLOUTS,
          BDIM_X,
          BDIM_Y>::control_trajectory;

  using state_trajectory = typename Controller<DYN_T, COST_T,
          MAX_TIMESTEPS,
          NUM_ROLLOUTS,
          BDIM_X,
          BDIM_Y>::state_trajectory;

  using state_array = typename Controller<DYN_T, COST_T,
          MAX_TIMESTEPS,
          NUM_ROLLOUTS,
          BDIM_X,
          BDIM_Y>::state_array;

  using sampled_cost_traj = typename Controller<DYN_T, COST_T,
          MAX_TIMESTEPS,
          NUM_ROLLOUTS,
          BDIM_X,
          BDIM_Y>::sampled_cost_traj;

  using FeedbackGainTrajectory = typename util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM>;
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

  TubeMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter, float gamma,
                     const Eigen::Ref<const StateCostWeight>& Q,
                     const Eigen::Ref<const Hessian>& Qf,
                     const Eigen::Ref<const ControlCostWeight>& R,
                     const Eigen::Ref<const control_array>& control_variance,
                     int num_timesteps = MAX_TIMESTEPS,
                     const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                     cudaStream_t stream = nullptr);

//  TubeMPPIController() = default;

  void computeControl(const Eigen::Ref<const state_array>& state) override;

  /**
   * returns the current control sequence
   */
  control_trajectory getNominalControlSeq() { return nominal_control_trajectory_;};

  /**
   * returns the current state sequence
   */
  state_trajectory getNominalStateSeq() {return nominal_state_trajectory_;};

  /**
   * Slide the control sequence back n steps
   */
  void slideControlSequence(int steps) override;

  void initDDP(const StateCostWeight& q_mat,
               const Hessian& q_f_mat,
               const ControlCostWeight& r_mat);

  void computeFeedbackGains(const Eigen::Ref<const state_array>& s) override;

  FeedbackGainTrajectory getFeedbackGains() override { return result_.feedback_gain;};

  state_trajectory getAncillaryStateSeq() {return result_.state_trajectory;};

  void smoothControlTrajectory();

  void updateNominalState(const Eigen::Ref<const control_array>& u);

  StateCostWeight Q_;
  Hessian Qf_;
  ControlCostWeight R_;

  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;

private:
  float normalizer_nominal_; // Variable for the normalizing term from sampling.
  float baseline_nominal_; // Baseline cost of the system.
  float nominal_threshold_ = 100; // How much worse the actual system has to be compared to the nominal

  float* initial_state_nominal_d_; // Array of sizae DYN_T::STATE_DIM * (2 if there is a nominal state)

  control_trajectory nominal_control_trajectory_ = control_trajectory::Zero();
  state_trajectory nominal_state_trajectory_ = state_trajectory::Zero();

  // for DDP
  control_array control_min_;
  control_array control_max_;

  OptimizerResult<ModelWrapperDDP<DYN_T>> result_;

  sampled_cost_traj trajectory_costs_nominal_ = sampled_cost_traj::Zero();

  // Check to see if nominal state has been initialized
  bool nominalStateInit_ = false;

  void computeStateTrajectory(const Eigen::Ref<const state_array>& x0_actual);

protected:

  // TODO move up and generalize, pass in what to copy and initial location
  void copyControlToDevice();

private:
  // ======== PURE VIRTUAL =========
  void allocateCUDAMemory();
  // ======== PURE VIRTUAL END =====
};


#if __CUDACC__
#include "tube_mppi_controller.cu"
#endif

#endif
