#include <controllers/cartpole_mppi.cuh>

/*
 * This file contains the instantiations of the controller for the cart pole.
 * Will have a dynamics model of cartpole, some cost function,
 * and a controller of just MPPI, (not tube or R)
 */

template class VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 150, 64, 8>;
