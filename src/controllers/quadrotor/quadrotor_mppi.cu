#include <mppi/instantiations/quadrotor_mppi/quadrotor_mppi.cuh>

template class VanillaMPPIController<QuadrotorDynamics, QuadrotorQuadraticCost, DDPFeedback<QuadrotorDynamics, 100>, 100, 512, 64, 8>;
template class VanillaMPPIController<QuadrotorDynamics, QuadrotorMapCost, DDPFeedback<QuadrotorDynamics, 100>, 100, 512, 64, 8>;
