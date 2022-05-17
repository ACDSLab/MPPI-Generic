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
    ROLL, // TODO delete
    PITCH,
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    BRAKE_THROTTLE = 0,
    DESIRED_STEERING,
    NUM_CONTROLS
  };

  enum class OutputIndex : int
  {
    VEL_X = 0,
    YAW,
    POS_X,
    POS_Y,
    TRUE_STEER_ANGLE,
    ROLL,
    PITCH,
    NUM_OUTPUTS
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
template <class CLASS_T>
class RacerDubinsImpl : public Dynamics<CLASS_T, RacerDubinsParams, 7, 2> // TODO should be 5
{
public:
  typedef Dynamics<CLASS_T, RacerDubinsParams, 7, 2> PARENT_CLASS;
  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;

  // TODO use new enum for this
  static const int CTRL_THROTTLE_BRAKE = 0;
  static const int CTRL_STEER_CMD = 1;

  static const int STATE_V = 0;
  static const int STATE_YAW = 1;
  static const int STATE_PX = 2;
  static const int STATE_PY = 3;
  static const int STATE_STEER = 4;
  // static const int STATE_NB_DYNAMICS_STATES = 5;
  // outputs in state vector
  // static const int STATE_OUT_STEER = STATE_STEER; // ideal Ackerman in radian
  // static const int STATE_OUT_STEER_VEL = 5; // in rad/s
  // static const int STATE_OUT_BASELINK_POS_I_X = STATE_PX; // base_link pos in inertial
  // static const int STATE_OUT_BASELINK_POS_I_Y = STATE_PY;
  // static const int STATE_OUT_BASELINK_POS_I_Z = 6;
  // static const int STATE_OUT_BASELINK_VEL_B_X = STATE_V; // base_link vel in body
  // static const int STATE_OUT_BASELINK_VEL_B_Y = 7;
  // static const int STATE_OUT_BASELINK_VEL_B_Z = 8;
  // static const int STATE_OUT_OMEGA_B_X = 9; // in body
  // static const int STATE_OUT_OMEGA_B_Y = 10;
  // static const int STATE_OUT_OMEGA_B_Z = 11;
  // static const int STATE_OUT_ATTITUDE_QW = 12; // body to inertial
  // static const int STATE_OUT_ATTITUDE_QX = 13;
  // static const int STATE_OUT_ATTITUDE_QY = 14;
  // static const int STATE_OUT_ATTITUDE_QZ = 15;
  // static const int STATE_OUT_WHEEL_POS_FL_FR_RL_RR_XY = 16; // to 23, in inertial
  // static const int STATE_OUT_WHEEL_POS_FL_X = 16;
  // static const int STATE_OUT_WHEEL_POS_FL_Y = 17;
  // static const int STATE_OUT_WHEEL_POS_FR_X = 18;
  // static const int STATE_OUT_WHEEL_POS_FR_Y = 19;
  // static const int STATE_OUT_WHEEL_POS_RL_X = 20;
  // static const int STATE_OUT_WHEEL_POS_RL_Y = 21;
  // static const int STATE_OUT_WHEEL_POS_RR_X = 22;
  // static const int STATE_OUT_WHEEL_POS_RR_Y = 23;
  // static const int STATE_OUT_WHEEL_FORCE_FL_FR_RL_RR_XYZ = 24; // to 36, in body
  // static const int STATE_OUT_WHEEL_FORCE_FL_X = 25;
  // static const int STATE_OUT_WHEEL_FORCE_FL_Y = 26;
  // static const int STATE_OUT_WHEEL_FORCE_FL_Z = 27;
  // static const int STATE_OUT_WHEEL_FORCE_FR_X = 28;
  // static const int STATE_OUT_WHEEL_FORCE_FR_Y = 29;
  // static const int STATE_OUT_WHEEL_FORCE_FR_Z = 30;
  // static const int STATE_OUT_WHEEL_FORCE_RL_X = 31;
  // static const int STATE_OUT_WHEEL_FORCE_RL_Y = 32;
  // static const int STATE_OUT_WHEEL_FORCE_RL_Z = 33;
  // static const int STATE_OUT_WHEEL_FORCE_RR_X = 34;
  // static const int STATE_OUT_WHEEL_FORCE_RR_Y = 35;
  // static const int STATE_OUT_WHEEL_FORCE_RR_Z = 36;
  // static const int STATE_OUT_CENTER_POS_X = 37; // in inertial
  // static const int STATE_OUT_CENTER_POS_Y = 38;
  // static const int STATE_OUT_CENTER_POS_Z = 39;
  // static const int STATE_NB_OUTPUT_STATES = STATE_DIM - STATE_NB_DYNAMICS_STATES;


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

class RacerDubins : public RacerDubinsImpl<RacerDubins>
{
public:
  RacerDubins(cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubins>(stream)
  {
  }

  RacerDubins(RacerDubinsParams& params, cudaStream_t stream = nullptr)
    : RacerDubinsImpl<RacerDubins>(params, stream)
  {
  }
};

#if __CUDACC__
#include "racer_dubins.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_CUH
