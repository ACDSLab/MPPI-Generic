#ifndef AR_NN_DYNAMICS_CUH_
#define AR_NN_DYNAMICS_CUH_

#include <dynamics/dynamics.cuh>

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
class NeuralNetModel : public Dynamics<S_DIM, C_DIM> {

};

#if __CUDACC__
#include "ar_standard_cost.cu"
#endif

#endif AR_NN_DYNAMICS_CUH_