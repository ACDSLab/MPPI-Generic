/**
 * Created by jason on 10/30/19.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
**/

#ifndef MPPIGENERIC_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_MPPI_CONTROLLER_CUH

#include "mppi_common.cuh"

// Double check if these are included in mppi_common.h
#include <Eigen/Dense>
#include <vector>

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
class VanillaMPPIController {
public:
    DYN_T model_;
    COST_T cost_;

    VanillaMPPIController(DYN_T model, COST_T cost) : model_(model), cost_(cost) {};


	/**
	 * Aliases for control-related components
	 */
	typedef std::array<float, DYN_T::CONTROL_DIM> control_t; // Control at a time t
	typedef std::array<control_t, NUM_TIMESTEPS> control_trajectory; // A control trajectory
	// All control trajectories sampled
	typedef std::array<control_trajectory, NUM_ROLLOUTS> sampled_control_traj;

	/**
	 * Aliases for state-related components
	 */
	typedef std::array<float, DYN_T::STATE_DIM> state_t; // State at a time t
	typedef std::array<state_t, NUM_TIMESTEPS> state_trajectory; // A state trajectory
	 // All state trajectories sampled
	typedef std::array<state_trajectory, NUM_ROLLOUTS> sampled_state_traj;

	/**
	 * Aliases for control-related components
	 */
	typedef std::array<float, NUM_TIMESTEPS> cost_trajectory; // A cost trajectory
	 // All cost trajectories sampled
	typedef std::array<cost_trajectory, NUM_ROLLOUTS> sampled_cost_traj;
//	/**
//	 * Given a state, calculates the optimal control sequence using MPPI according
//	 * to the cost function used as part of the template
//	 * @param state - the current state from which we would like to calculate
//	 * a control sequence
//	 */
//	void computeControl(Eigen::Matrix<float, DYN_T::STATE_DIM,1> state);

	/**
	 * Given a state, calculates the optimal control sequence using MPPI according
	 * to the cost function used as part of the template
	 * @param state - the current state from which we would like to calculate
	 * a control sequence
	 *
	 * returns the control sequence after computation
	 */
	control_trajectory computeControl(Eigen::Matrix<float, DYN_T::STATE_DIM, 1> state);

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


};

#if __CUDACC__
#include "mppi_controller.cu"
#endif

#endif //MPPIGENERIC_MPPI_CONTROLLER_CUH
