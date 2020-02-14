/**
 * Created by jason on 10/30/19.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
**/

#ifndef MPPIGENERIC_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_MPPI_CONTROLLER_CUH

#include "curand.h"
// Double check if these are included in mppi_common.h
#include <chrono>

#include <controllers/controller.cuh>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
class VanillaMPPIController : public Controller<DYN_T, COST_T,
                                                MAX_TIMESTEPS,
                                                NUM_ROLLOUTS,
                                                BDIM_X,
                                                BDIM_Y> {
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
    /**
     *
     * Public member functions
     */
    // Constructor
    VanillaMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                          float gamma, int num_timesteps,
                          const control_array& control_variance,
                          const control_trajectory& init_control_traj = control_trajectory(),
                          cudaStream_t stream= nullptr);

    // Destructor
    ~VanillaMPPIController();


    void computeNominalStateTrajectory(const state_array& x0);

    void updateControlNoiseVariance(const control_array& sigma_u);

    control_array getControlVariance() { return control_variance_;};

    float getBaselineCost() {return baseline_;};

    virtual void computeControl(state_array state) override;

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

    cudaStream_t stream_;

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
    sampled_cost_traj trajectory_costs_ = {{0}};
    control_array control_variance_ = {{0}};
    control_trajectory control_history_ = {{0}};

    float* initial_state_d_;
    float* nominal_control_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS
    float* nominal_state_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS
    float* trajectory_costs_d_; // Array of size NUM_ROLLOUTS
    float* control_variance_d_; // Array of size DYN_T::CONTROL_DIM
    float* control_noise_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS

    // WARNING This method is private because it is only called once in the constructor. Logic is required
    // so that CUDA memory is properly reallocated when the number of timesteps changes.
    void setNumTimesteps(int num_timesteps);

    void createAndSeedCUDARandomNumberGen();

    void setCUDAStream(cudaStream_t stream);

    // Allocate CUDA memory for the controller
    void allocateCUDAMemory();

    // Free CUDA memory for the controller
    void deallocateCUDAMemory();

    void copyControlVarianceToDevice();

    void copyNominalControlToDevice();

};


#if __CUDACC__
#include "mppi_controller.cu"
#endif

#endif //MPPIGENERIC_MPPI_CONTROLLER_CUH
