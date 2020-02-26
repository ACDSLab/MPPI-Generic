#include <instantiations/cartpole_mppi/cartpole_mppi.cuh>

/*
 * This file contains the instantiations of the controller for the cart pole.
 * Will have a dynamics model of cartpole, some cost function,
 * and a controller of just MPPI, (not tube or R)
 */
// Num_timesteps, num_rollouts, blockdim x, blockdim y
template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>;
template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 256, 64, 8>;
template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 150, 512, 64, 8>;



template class TubeMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>;
