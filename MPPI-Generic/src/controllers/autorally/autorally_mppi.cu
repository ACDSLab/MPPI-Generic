#include <instantiations/autorally_mppi/autorally_mppi.cuh>

/*
 * This file contains the instantiations of the controller for the cart pole.
 * Will have a dynamics model of cartpole, some cost function,
 * and a controller of just MPPI, (not tube or R)
 */

//Convenience typedef for the MPPI Controller.
template class ARStandardCost<>;
template class VanillaMPPIController<DynamicsModel, ARStandardCost<>, NUM_TIMESTEPS,
									 MPPI_NUM_ROLLOUTS__, BLOCKSIZE_X, BLOCKSIZE_Y>;

// template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 150, 64, 8>;
