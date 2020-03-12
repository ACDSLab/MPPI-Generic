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

//  using m_dyn = typename ModelWrapperDDP<DYN_T>::Scalar;
  using FeedbackGainTrajectory = typename util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM>;
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

  TubeMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                     float gamma, int num_timesteps,
                     const Eigen::Ref<const StateCostWeight>& Q,
                     const Eigen::Ref<const Hessian>& Qf,
                     const Eigen::Ref<const ControlCostWeight>& R,
                     const Eigen::Ref<const control_array>& control_variance,
                     const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                     cudaStream_t stream = nullptr);

  TubeMPPIController() = default;

  ~TubeMPPIController();

  void computeControl(const Eigen::Ref<const state_array>& state) override;

  /**
   * returns the current control sequence
   */
  control_trajectory getControlSeq() override { return nominal_control_trajectory;};

  /**
   * returns the current state sequence
   */
  state_trajectory getStateSeq() override {return nominal_state_trajectory;};

  /**
   * Slide the control sequence back n steps
   */
  void slideControlSequence(int steps) override;

  void updateControlNoiseVariance(const Eigen::Ref<const control_array>& sigma_u);


  void initDDP(const StateCostWeight& q_mat,
               const Hessian& q_f_mat,
               const ControlCostWeight& r_mat);

  void computeFeedbackGains(const Eigen::Ref<const state_array>& s) override;

  FeedbackGainTrajectory getFeedbackGains() { return result_.feedback_gain;};

  StateCostWeight Q_;
  Hessian Qf_;
  ControlCostWeight R_;

  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;
  cudaStream_t stream_;

private:
  int num_iters_;  // Number of optimization iterations
  float gamma_; // Value of the temperature in the softmax.
  float normalizer_actual_; // Variable for the normalizing term from sampling.
  float normalizer_nominal_; // Variable for the normalizing term from sampling.
  float baseline_actual_; // Baseline cost of the system.
  float baseline_nominal_; // Baseline cost of the system.
  float dt_;
  float nominal_threshold_; // How much worse the actual system has to be compared to the nominal

  control_trajectory nominal_control_trajectory = control_trajectory::Zero();
  control_trajectory actual_control_trajectory = control_trajectory::Zero();
  state_trajectory nominal_state_trajectory = state_trajectory::Zero();
  state_trajectory actual_state_trajectory = state_trajectory::Zero();


  control_array control_min_;
  control_array control_max_;

  OptimizerResult<ModelWrapperDDP<DYN_T>> result_;

  sampled_cost_traj trajectory_costs_nominal_ = sampled_cost_traj::Zero();
  sampled_cost_traj trajectory_costs_actual_ = sampled_cost_traj::Zero();

  // Check to see if nominal state has been initialized
  bool nominalStateInit_ = false;
  bool use_nominal_state_ = false;

  float* initial_state_d_;
  // Each of these device arrays are twice as large as in MPPI to hold
  // both the nominal and actual values.
  float* control_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS * 2
  float* state_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS * 2
  float* trajectory_costs_d_; // Array of size NUM_ROLLOUTS * 2
  float* control_noise_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS * 2



  void computeNominalStateTrajectory(const Eigen::Ref<const state_array>& x0);

protected:
  int num_timesteps_;
  curandGenerator_t gen_;

  control_array control_variance_ = control_array::Zero();
  float* control_variance_d_;

  // WARNING This method is private because it is only called once in the constructor. Logic is required
  // so that CUDA memory is properly reallocated when the number of timesteps changes.
  void setNumTimesteps(int num_timesteps);

  void createAndSeedCUDARandomNumberGen();

  void setCUDAStream(cudaStream_t stream);


  // Allocate CUDA memory for the controller
  void allocateCUDAMemory();

  // Free CUDA memory for the controller
  void deallocateCUDAMemory();

  void copyControlToDevice();

  void copyControlVarianceToDevice();
};


#if __CUDACC__
#include "tube_mppi_controller.cu"
#endif

#endif
