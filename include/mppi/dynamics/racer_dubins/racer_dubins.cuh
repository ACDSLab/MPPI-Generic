#ifndef MPPIGENERIC_RACER_DUBINS_CUH
#define MPPIGENERIC_RACER_DUBINS_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/angle_utils.cuh>

struct RacerDubinsParams
{
  float c_t = 1.3;
  float c_b = 2.5;
  float c_v = 3.7;
  float c_0 = 4.9;
  float steering_constant = .6;
  float wheel_base = 0.3;
  float steer_command_angle_scale = -2.45;
  float gravity = 0.0;
};

using namespace MPPI_internal;
/**
 * state: v, theta, p_x, p_y, true steering angle
 * control: throttle, steering angle command
 */
template<class CLASS_T, int STATE_DIM>
class RacerDubinsImpl : public Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>
{
public:
  typedef typename Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>::state_array state_array;
  typedef typename Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>::control_array control_array;
  typedef typename Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>::dfdx dfdx;
  typedef typename Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>::dfdu dfdu;

  RacerDubinsImpl(cudaStream_t stream = nullptr) : Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>(stream)
  {
  }
  RacerDubinsImpl(RacerDubinsParams& params, cudaStream_t stream = nullptr)
    : Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>(params, stream)
  {
  }

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  void updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt);

  __device__ void updateState(float* state, float* state_der, const float dt);

  state_array interpolateState(const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2,
                               const float alpha);

  bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);

  void getStoppingControl(const Eigen::Ref<const state_array>& state, Eigen::Ref<control_array> u);

};

class RacerDubins : public RacerDubinsImpl<RacerDubins, 5>
{
public:
  RacerDubins(cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubins, 5>(stream)
  {}

  RacerDubins(RacerDubinsParams& params, cudaStream_t stream = nullptr)
  : RacerDubinsImpl<RacerDubins, 5>(params, stream)
  {}
};

#if __CUDACC__
#include "racer_dubins.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_CUH
