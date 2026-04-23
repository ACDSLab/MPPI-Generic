#include <mppi/instantiations/double_integrator_mppi/double_integrator_mppi.cuh>

/*
 * This file contains the instantiations of the controller for the cart pole.
 * Will have a dynamics model of cartpole, some cost function,
 * and a controller of just MPPI, (not tube or R)
 */
// Num_timesteps, num_rollouts, blockdim x, blockdim y
typedef mppi::sampling_distributions::GaussianDistribution<DoubleIntegratorDynamics::DYN_PARAMS_T> Sampler;

template class VanillaMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost,
                                     DDPFeedback<DoubleIntegratorDynamics>, Sampler>;

template class TubeMPPIController<DoubleIntegratorDynamics, DoubleIntegratorCircleCost,
                                  DDPFeedback<DoubleIntegratorDynamics>, Sampler>;
