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
  float c_t = 1.3;
  float c_b = 2.5;
  float c_v = 3.7;
  float c_0 = 4.9;
  float steering_constant = .6;
  float wheel_base = 0.3;
  float steer_command_angle_scale = -2.45;
  float gravity = -9.81;
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

  static const int CTRL_THROTTLE_BRAKE = 0;
  static const int CTRL_STEER_CMD = 1;
  static const int STATE_V = 0;
  static const int STATE_YAW = 1;
  static const int STATE_PX = 2;
  static const int STATE_PY = 3;
  static const int STATE_STEER = 4;

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

  Eigen::Quaternionf attitudeFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f positionFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f velocityFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f angularRateFromState(const Eigen::Ref<const state_array>& state);
  state_array stateFromOdometry(const Eigen::Quaternionf &q, const Eigen::Vector3f &pos, const Eigen::Vector3f &vel, const Eigen::Vector3f &omega);
};

class RacerDubins : public RacerDubinsImpl<RacerDubins, 5>
{
public:
  RacerDubins(cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubins, 5>(stream)
  {
  }

  RacerDubins(RacerDubinsParams& params, cudaStream_t stream = nullptr)
    : RacerDubinsImpl<RacerDubins, 5>(params, stream)
  {
  }
};

#if __CUDACC__
#include "racer_dubins.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_CUH
