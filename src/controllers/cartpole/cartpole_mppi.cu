#include <mppi/instantiations/cartpole_mppi/cartpole_mppi.cuh>

/*
 * This file contains the instantiations of the controller for the cart pole.
 * Will have a dynamics model of cartpole, some cost function,
 * and a controller of just MPPI, (not tube or R)
 */
// Num_timesteps, num_rollouts, blockdim x, blockdim y
const int NUMBER_TIMESTEPS = 100;
template class GPUFeedbackController<DeviceDDPImpl<DeviceDDP<CartpoleDynamics>, CartpoleDynamics>, CartpoleDynamics>;
template class DeviceDDPImpl<DeviceDDP<CartpoleDynamics>, CartpoleDynamics>;
template class DeviceDDP<CartpoleDynamics>;
template class DDPFeedback<CartpoleDynamics, 150>;

template class DDPFeedback<CartpoleDynamics, NUMBER_TIMESTEPS>;
template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, DDPFeedback<CartpoleDynamics, NUMBER_TIMESTEPS>, NUMBER_TIMESTEPS, 2048, 64, 8>;
template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, DDPFeedback<CartpoleDynamics, NUMBER_TIMESTEPS>, NUMBER_TIMESTEPS, 256, 64, 8>;
template class VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, DDPFeedback<CartpoleDynamics, 150>, 150, 512, 64, 8>;



template class TubeMPPIController<CartpoleDynamics, CartpoleQuadraticCost, DDPFeedback<CartpoleDynamics, NUMBER_TIMESTEPS>, NUMBER_TIMESTEPS, 2048, 64, 8>;
