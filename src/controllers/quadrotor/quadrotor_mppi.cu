#include <mppi/instantiations/quadrotor_mppi/quadrotor_mppi.cuh>

template class VanillaMPPIController<QuadrotorDynamics, QuadrotorQuadraticCost, 100, 512, 64, 8>;