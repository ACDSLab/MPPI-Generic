/**
  * Created by Manan Gandhi on 2/14/2020
  * API for interfacing with the TUBE MPPI controller.
  */

#ifndef MPPIGENERIC_TUBE_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_TUBE_MPPI_CONTROLLER_CUH

#include <curand.h>
#include <controllers/controller.cuh>
#include <ddp/ddp_model_wrapper.h>
#include <ddp/ddp_tracking_costs.h>
#include <ddp/ddp.h>
#include <eigen3/Eigen/Dense>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y>
class TubeMPPIController: public Controller<DYN_T, COST_T, MAX_TIMESTEPS,
                                            NUM_ROLLOUTS, BDIM_X, BDIM_Y> {

public:
    using typename Controller<DYN_T, COST_T,
            MAX_TIMESTEPS,
            NUM_ROLLOUTS,
            BDIM_X,
            BDIM_Y>::control_array;

    using typename Controller<DYN_T, COST_T,
            MAX_TIMESTEPS,
            NUM_ROLLOUTS,
            BDIM_X,
            BDIM_Y>::control_trajectory;

    using typename Controller<DYN_T, COST_T,
            MAX_TIMESTEPS,
            NUM_ROLLOUTS,
            BDIM_X,
            BDIM_Y>::state_trajectory;

    using typename Controller<DYN_T, COST_T,
            MAX_TIMESTEPS,
            NUM_ROLLOUTS,
            BDIM_X,
            BDIM_Y>::state_array;

    using typename Controller<DYN_T, COST_T,
            MAX_TIMESTEPS,
            NUM_ROLLOUTS,
            BDIM_X,
            BDIM_Y>::sampled_cost_traj;

    TubeMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                       float gamma, int num_timesteps,
                       const control_array& control_variance,
                       const control_trajectory& init_control_traj = control_trajectory(),
                       cudaStream_t stream= nullptr);

    void computeControl(state_array state) override {};

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
    void slideControlSequence(int steps) override {};

private:
    int num_iters_;  // Number of optimization iterations
    int num_timesteps_;
    float gamma_; // Value of the temperature in the softmax.
    float normalizer_; // Variable for the normalizing term from sampling.
    float baseline_; // Baseline cost of the system.
    float dt_;


    curandGenerator_t gen_;

    control_trajectory nominal_control_ = {{0}};
    state_trajectory nominal_state_ = {{0}};

};


#if __CUDACC__
#include "tube_mppi_controller.cu"
#endif

#endif