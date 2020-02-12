/**
 * Created by jason on 10/30/19.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
**/

#ifndef MPPIGENERIC_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_MPPI_CONTROLLER_CUH

#include "mppi_common.cuh"
#include "curand.h"
// Double check if these are included in mppi_common.h
#include <Eigen/Dense>
#include <vector>



template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
class VanillaMPPIController {
public:
	/**
	 * Aliases
	 */
	 // Control
    typedef std::array<float, DYN_T::CONTROL_DIM> control_array; // State at a time t
    typedef std::array<float, DYN_T::CONTROL_DIM*NUM_TIMESTEPS> control_trajectory; // A control trajectory
	typedef std::array<float, DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS> sampled_control_traj; // All control trajectories sampled
    // State
	typedef std::array<float, DYN_T::STATE_DIM> state_array; // State at a time t
	typedef std::array<float, DYN_T::STATE_DIM*NUM_TIMESTEPS> state_trajectory; // A state trajectory
	typedef std::array<float, DYN_T::STATE_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS> sampled_state_traj; // All state trajectories sampled
    // Cost
	typedef std::array<float, NUM_TIMESTEPS> cost_trajectory; // A cost trajectory
	typedef std::array<cost_trajectory, NUM_ROLLOUTS> sampled_cost_traj; // All cost trajectories sampled

	/**
	 * Public data members
	 */
	DYN_T* model_;
    COST_T* cost_;



    /**
     * Public member functions
     */
     // Constructor
    VanillaMPPIController(DYN_T* model, COST_T* cost, float dt, cudaStream_t stream);

    // Destructor
    ~VanillaMPPIController();

    // CUDA memory management
    void GPUSetup();

    // Initialize the control sequence
    void initControlTrajectory(control_trajectory init_control);


    /**
     * Given a state, calculates the optimal control sequence using MPPI according
     * to the cost function used as part of the template
     * @param state - the current state from which we would like to calculate
     * a control sequence
     */
    void computeControl(state_array state);

    /**
     * returns the current control sequence
     */
    control_trajectory get_control_seq();

    /**
     * return the entire sample of control sequences
     */
    sampled_control_traj get_sampled_control_seq();

    /**
     * Return the current optimal state sequence
     */
    state_trajectory get_state_seq();

    /**
     * Return all the sampled states sequences
     */
    sampled_state_traj get_sampled_state_seq();

    /**
     * Return the current minimal cost sequence
     */
    cost_trajectory get_cost_seq();

    /**
     * Return all the sampled costs sequences
     */
    sampled_cost_traj get_sampled_cost_seq();

    cudaStream_t stream_;


private:
    int num_iters_;  // Number of optimization iterations
    float gamma_; // Value of the temperature in the softmax.
    float normalizer_; // Variable for the normalizing term from sampling.
    float dt_;


    curandGenerator_t gen_;

    control_trajectory nominal_control_;
    state_trajectory nominal_state_;
    sampled_cost_traj trajectory_costs_;
    control_array control_variance_;
    control_trajectory control_history_;

    float* nominal_control_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS
    float* nominal_state_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS
    float* trajectory_costs_d_; // Array of size NUM_ROLLOUTS
    float* control_variance_d_; // Array of size DYN_T::CONTROL_DIM
    float* control_noise_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS

    void setCUDAStream(cudaStream_t stream);

    // Allocate CUDA memory for the controller
    void allocateCUDAMemory();

    // Free CUDA memory for the controller
    void deallocateCUDAMemory();

};


#if __CUDACC__
#include "mppi_controller.cu"
#endif

#endif //MPPIGENERIC_MPPI_CONTROLLER_CUH
