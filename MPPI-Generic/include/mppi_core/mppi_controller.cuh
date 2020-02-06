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

// TODO Figure out template here
class MPPIController {
public:
	const int STATE_DIM = 1; // TODO: placeholder, Replace with template
	/**
	 * Given a state, calculates the optimal control sequence using MPPI according
	 * to the cost function used as part of the template
	 * @param state - the current state from which we would like to calculate
	 * a control sequence
	 */
	void compute_control(Eigen::Matrix<float, STATE_DIM,1> state);

	/**
	 * Given a state, calculates the optimal control sequence using MPPI according
	 * to the cost function used as part of the template
	 * @param state - the current state from which we would like to calculate
	 * a control sequence
	 *
	 * returns the control sequence after computation
	 */
	std::vector<float> compute_control(Eigen::<float, STATE_DIM, 1> state);

	/**
	 * returns the current control sequence
	 */
	std::vector<float> get_control_seq();

};

#if __CUDACC__
#include "mppi_controller.cu"
#endif

#endif //MPPIGENERIC_MPPI_CONTROLLER_CUH
