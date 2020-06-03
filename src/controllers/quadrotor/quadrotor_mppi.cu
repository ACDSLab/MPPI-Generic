#include <mppi/instantiations/quadrotor_mppi/quadrotor_mppi.cuh>

template class VanillaMPPIController<QuadrotorDynamics, DoubleIntegratorCircleCost, 100, 512, 64, 8>;