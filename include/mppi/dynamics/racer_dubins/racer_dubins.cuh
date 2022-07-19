#ifndef MPPIGENERIC_RACER_DUBINS_CUH
#define MPPIGENERIC_RACER_DUBINS_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/angle_utils.cuh>

struct RacerDubinsParams : public DynamicsParams
{
  enum class StateIndex : int
  {
    VEL_X = 0,
    YAW,
    POS_X,
    POS_Y,
    TRUE_STEER_ANGLE,
    ROLL,
    PITCH,
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    BRAKE_THROTTLE = 0,
    DESIRED_STEERING,
    NUM_CONTROLS
  };
  float c_t[3] = {1.3, 2.6, 3.9};
  float c_b[3] = {2.5, 3.5, 4.5};
  float c_v[3] = {3.7, 4.7, 5.7};
  float c_0 = 4.9;
  float steering_constant = .6;
  float wheel_base = 0.3;
  float steer_command_angle_scale = 5;
  float steer_angle_scale[3] = {-9.1, -10.2, -15.1};
  float low_min_throttle = 0.13;
  float gravity = -9.81;
  float max_steer_angle = 0.5;
  float max_steer_rate = 5;
};

using namespace MPPI_internal;
/**
 * state: v, theta, p_x, p_y, true steering angle
 * control: throttle, steering angle command
 */
template <class CLASS_T, int STATE_DIM>
class RacerDubinsImpl : public Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2>
{
public:
  typedef Dynamics<CLASS_T, RacerDubinsParams, STATE_DIM, 2> PARENT_CLASS;
  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;

  RacerDubinsImpl(cudaStream_t stream = nullptr) : PARENT_CLASS(stream)
  {
  }
  RacerDubinsImpl(RacerDubinsParams& params, cudaStream_t stream = nullptr) : PARENT_CLASS(params, stream)
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
  Eigen::Quaternionf get_attitude(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f get_position(const Eigen::Ref<const state_array>& state);

  void enforceLeash(const Eigen::Ref<const state_array>& state_true, const Eigen::Ref<const state_array>& state_nominal, const Eigen::Ref<const state_array>& leash_values, Eigen::Ref<state_array> state_output);
};

class RacerDubins : public RacerDubinsImpl<RacerDubins, 7>
{
public:
  RacerDubins(cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubins, 7>(stream)
  {
  }

  RacerDubins(RacerDubinsParams& params, cudaStream_t stream = nullptr)
    : RacerDubinsImpl<RacerDubins, 7>(params, stream)
  {
  }
};

#if __CUDACC__
#include "racer_dubins.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_CUH
