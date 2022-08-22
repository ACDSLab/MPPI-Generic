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
    STEER_ANGLE,
    STEER_ANGLE_RATE,
    ACCEL_X,
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    THROTTLE_BRAKE = 0,
    STEER_CMD,
    NUM_CONTROLS
  };

  enum class OutputIndex : int
  {
    BASELINK_VEL_B_X = 0,
    BASELINK_VEL_B_Y,
    BASELINK_VEL_B_Z,
    BASELINK_POS_I_X,
    BASELINK_POS_I_Y,
    BASELINK_POS_I_Z,
    YAW,
    ROLL,
    PITCH,
    STEER_ANGLE,
    STEER_ANGLE_RATE,
    WHEEL_POS_I_FL_X,
    WHEEL_POS_I_FL_Y,
    WHEEL_POS_I_FR_X,
    WHEEL_POS_I_FR_Y,
    WHEEL_POS_I_RL_X,
    WHEEL_POS_I_RL_Y,
    WHEEL_POS_I_RR_X,
    WHEEL_POS_I_RR_Y,
    WHEEL_FORCE_B_FL,
    WHEEL_FORCE_B_FR,
    WHEEL_FORCE_B_RL,
    WHEEL_FORCE_B_RR,
    ACCEL_X,
    NUM_OUTPUTS
  };

  float c_t[3] = { 1.3, 2.6, 3.9 };
  float c_b[3] = { 2.5, 3.5, 4.5 };
  float c_v[3] = { 3.7, 4.7, 5.7 };
  float c_0 = 4.9;
  float steering_constant = .6;
  float wheel_base = 0.3;
  float steer_command_angle_scale = 5;
  float steer_angle_scale[3] = { -9.1, -10.2, -15.1 };
  float low_min_throttle = 0.13;
  float gravity = -9.81;
  float max_steer_angle = 0.5;
  float max_steer_rate = 5;
};

using namespace MPPI_internal;

template <class CLASS_T, class PARAMS_T = RacerDubinsParams>
class RacerDubinsImpl : public Dynamics<CLASS_T, PARAMS_T>
{
public:
  typedef Dynamics<CLASS_T, PARAMS_T> PARENT_CLASS;
  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::output_array output_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;
  using PARENT_CLASS::updateState;  // needed as overloading updateState here hides all parent versions of updateState

  RacerDubinsImpl(cudaStream_t stream = nullptr) : PARENT_CLASS(stream)
  {
  }

  RacerDubinsImpl(PARAMS_T& params, cudaStream_t stream = nullptr) : PARENT_CLASS(params, stream)
  {
  }

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  // void computeStateDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
  //                         Eigen::Ref<state_array> state_der, output_array* output=nullptr); // TODO

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt);

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt);

  state_array interpolateState(const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2,
                               const float alpha);

  bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);

  // __device__ void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s, float
  // *output=nullptr); // TODO

  void getStoppingControl(const Eigen::Ref<const state_array>& state, Eigen::Ref<control_array> u);

  Eigen::Quaternionf attitudeFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f positionFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f velocityFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f angularRateFromState(const Eigen::Ref<const state_array>& state);
  state_array stateFromOdometry(const Eigen::Quaternionf& q, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel,
                                const Eigen::Vector3f& omega);

  void enforceLeash(const Eigen::Ref<const state_array>& state_true, const Eigen::Ref<const state_array>& state_nominal,
                    const Eigen::Ref<const state_array>& leash_values, Eigen::Ref<state_array> state_output);
};

class RacerDubins : public RacerDubinsImpl<RacerDubins>
{
public:
  RacerDubins(cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubins>(stream)
  {
  }

  RacerDubins(RacerDubinsParams& params, cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubins>(params, stream)
  {
  }
};

#if __CUDACC__
#include "racer_dubins.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_CUH
