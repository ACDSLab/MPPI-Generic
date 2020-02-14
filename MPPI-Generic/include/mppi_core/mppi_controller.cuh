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
#include <Eigen/Dense>
#include <chrono>


template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
class VanillaMPPIController {
public:
    /**
     * typedefs for access to templated class from outside classes
     */
    typedef DYN_T TEMPLATED_DYNAMICS;
    typedef COST_T TEMPLATED_COSTS;
	/**
	 * Aliases
	 */
	 // Control
    typedef std::array<float, DYN_T::CONTROL_DIM> control_array; // State at a time t
    typedef std::array<float, DYN_T::CONTROL_DIM*MAX_TIMESTEPS> control_trajectory; // A control trajectory
	typedef std::array<float, DYN_T::CONTROL_DIM*MAX_TIMESTEPS*NUM_ROLLOUTS> sampled_control_traj; // All control trajectories sampled
    typedef std::array<float, DYN_T::CONTROL_DIM * DYN_T::STATE_DIM> K_matrix;

    // State
	typedef std::array<float, DYN_T::STATE_DIM> state_array; // State at a time t
	typedef std::array<float, DYN_T::STATE_DIM*MAX_TIMESTEPS> state_trajectory; // A state trajectory
	typedef std::array<float, DYN_T::STATE_DIM*MAX_TIMESTEPS*NUM_ROLLOUTS> sampled_state_traj; // All state trajectories sampled
    // Cost
	typedef std::array<float, MAX_TIMESTEPS> cost_trajectory; // A cost trajectory
	typedef std::array<float, NUM_ROLLOUTS> sampled_cost_traj; // All costs sampled for all rollouts

	/**
	 * Public data members
	 */
	DYN_T* model_;
    COST_T* cost_;



    /**
     * Public member functions
     */
     // Constructor
     VanillaMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter, float gamma, int num_timesteps,
                           const control_array& control_variance, const control_trajectory& init_control_traj = control_trajectory(),
                           cudaStream_t stream= nullptr);

    // Destructor
    ~VanillaMPPIController();


    void computeNominalStateTrajectory(const state_array& x0);

    void updateControlNoiseVariance(const control_array& sigma_u);

    /**
     * Given a state, calculates the optimal control sequence using MPPI according
     * to the cost function used as part of the template
     * @param state - the current state from which we would like to calculate
     * a control sequence
     */
    void computeControl(state_array state);

    control_array getControlVariance() { return control_variance_;};

    void slideControlSequence(int steps);

    float getBaselineCost() {return baseline_;};

    /**
     * returns the current control sequence
     */
    control_trajectory get_control_seq() { return nominal_control_;};

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

    /**
     * Return control feedback gains
     */
    K_matrix getFeedbackGains() {
        K_matrix empty_feedback_gain = {{0}};
        return empty_feedback_gain;
    };

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
