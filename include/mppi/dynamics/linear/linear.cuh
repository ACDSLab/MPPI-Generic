#ifndef MPPIGENERIC_LINEAR_CUH
#define MPPIGENERIC_LINEAR_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/angle_utils.cuh>

struct LinearModelParams
{
  float c_t = 1.3;
  float c_b = 2.5;
  float c_v = 3.7;
  float c_0 = 4.9;
  float wheel_base = 0.3;
};

using namespace MPPI_internal;
/**
 * state: v, theta, p_x, p_y
 * control: throttle, brake, gear selector, steering angle
 */
class LinearModel : public Dynamics<LinearModel, LinearModelParams, 4, 4>
{
public:
  LinearModel(cudaStream_t stream = nullptr) : Dynamics<LinearModel, LinearModelParams, 4, 4>(stream)
  {
  }
  LinearModel(LinearModelParams& params, cudaStream_t stream = nullptr)
    : Dynamics<LinearModel, LinearModelParams, 4, 4>(params, stream)
  {
  }

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  void updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt);

  __device__ void updateState(float* state, float* state_der, const float dt);

  state_array interpolateState(const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2,
                               const double alpha);

  bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);
};

#if __CUDACC__
#include "linear.cu"
#endif

#endif  // MPPIGENERIC_LINEAR_CUH
