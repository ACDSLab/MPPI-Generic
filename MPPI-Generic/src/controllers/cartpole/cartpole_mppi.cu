#include <instantiations/cartpole_mppi/cartpole_mppi.cuh>

/*
 * This file contains the instantiations of the controller for the cart pole.
 * Will have a dynamics model of cartpole, some cost function,
 * and a controller of just MPPI, (not tube or R)
 */
// Num_timesteps, num_rollouts, blockdim x, blockdim y
<<<<<<< 535e4e70bf73d21d48df2314e1304eefb66f18bd
template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, 100, 2048, 64, 8>;
=======


template class VanillaMPPIController<Cartpole, CartpoleQuadraticCost, 100, 2048, 64, 8>;

template class TubeMPPIController<Cartpole, CartpoleQuadraticCost, 100, 2048, 64, 8>;
>>>>>>> Created the base of the tube MPPI controller template. Added a compiling test.
