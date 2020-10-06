#include <mppi/instantiations/quadrotor_mppi/quadrotor_mppi.cuh>

template class VanillaMPPIController<QuadrotorDynamics, QuadrotorQuadraticCost, DDP<QuadrotorDynamics>, 100, 512, 64, 8>;
template class VanillaMPPIController<QuadrotorDynamics, QuadrotorMapCost, DDP<QuadrotorDynamics>, 100, 512, 64, 8>;
