#ifndef MPPIGENERIC_CONTROLLERERS_AUTORALLY_MPPI_CUH
#define MPPIGENERIC_CONTROLLERERS_AUTORALLY_MPPI_CUH

#include <dynamics/autorally/ar_nn_model.cuh>
#include <cost_functions/autorally/ar_standard_cost.cuh>
#include <mppi_core/mppi_controller.cuh>
#include <mppi_core/mppi_common.cuh>

// #ifdef USE_NEURAL_NETWORK_MODEL__ /*Use neural network dynamics model*/
const int MPPI_NUM_ROLLOUTS__ = 1920;
const int BLOCKSIZE_X = 8;
const int BLOCKSIZE_Y = 16;
const int NUM_TIMESTEPS = 150;
typedef NeuralNetModel<7,2,3,6,32,32,4> DynamicsModel;
typedef ARStandardCost<> CostFunction;
// #elif USE_BASIS_FUNC_MODEL__ /*Use the basis function model* */
// const int MPPI_NUM_ROLLOUTS__ = 2560;
// const int BLOCKSIZE_X = 16;
// const int BLOCKSIZE_Y = 4;
// typedef GeneralizedLinear<CarBasisFuncs, 7, 2, 25, CarKinematics, 3> DynamicsModel;
// #endif
#endif //MPPIGENERIC_CONTROLLERERS_AUTORALLY_MPPI_CUH
