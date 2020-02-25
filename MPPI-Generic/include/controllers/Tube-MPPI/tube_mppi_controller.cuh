/**
  * Created by Manan Gandhi on 2/14/2020
  * API for interfacing with the TUBE MPPI controller.
  */

#ifndef MPPIGENERIC_TUBE_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_TUBE_MPPI_CONTROLLER_CUH

#include <curand.h>
#include <controllers/controller.cuh>
#include <controllers/MPPI/mppi_controller.cuh>
#include <ddp/ddp_model_wrapper.h>
#include <ddp/ddp_tracking_costs.h>
#include <ddp/ddp.h>
#include <eigen3/Eigen/Dense>
#include <memory>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y>
class TubeMPPIController: public VanillaMPPIController<DYN_T, COST_T,
                                                       MAX_TIMESTEPS,
                                                       NUM_ROLLOUTS,
                                                       BDIM_X, BDIM_Y> {

public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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

    using typename Controller<DYN_T, COST_T,
            MAX_TIMESTEPS,
            NUM_ROLLOUTS,
            BDIM_X,
            BDIM_Y>::sampled_cost_traj;

    using m_dyn = typename ModelWrapperDDP<DYN_T>::Scalar;
    using FeedbackGainTrajectory = typename util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM>;
    using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
    using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
    using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

    TubeMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                       float gamma, int num_timesteps,
                       const StateCostWeight& Q,
                       const Hessian& Qf,
                       const ControlCostWeight& R,
                       const control_array& control_variance,
                       const control_trajectory& init_control_traj = control_trajectory(),
                       cudaStream_t stream= nullptr);

    void computeControl(const state_array& state) override;

    /**
     * returns the current control sequence
     */
    control_trajectory getControlSeq() override { return nominal_control_;};

    /**
     * returns the current state sequence
     */
    state_trajectory getStateSeq() override {return nominal_state_;};

    /**
     * Slide the control sequence back n steps
     */
    void slideControlSequence(int steps) override;

    void initDDP(const StateCostWeight& q_mat,
                 const Hessian& q_f_mat,
                 const ControlCostWeight& r_mat);

    void computeFeedbackGains(const state_array& s) override;

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

    control_trajectory nominal_control_ = control_trajectory::Zero();
    control_trajectory actual_control_ = control_trajectory::Zero();
    state_trajectory nominal_state_ = state_trajectory::Zero();
    state_trajectory actual_state_ = state_trajectory::Zero();

    control_array control_min_;
    control_array control_max_;

    OptimizerResult<ModelWrapperDDP<DYN_T>> result_;

    sampled_cost_traj trajectory_costs_nominal_ = {{0}};
    sampled_cost_traj trajectory_costs_actual_ = {{0}};

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
protected:
    // Allocate CUDA memory for the controller
    void allocateCUDAMemory() override;

    // Free CUDA memory for the controller
    void deallocateCUDAMemory() override;

    void copyControlToDevice();
};


#if __CUDACC__
#include "tube_mppi_controller.cu"
#endif

#endif
