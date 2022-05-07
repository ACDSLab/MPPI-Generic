#ifndef MPPIGENERIC_RACER_SUSPENSION_CUH
#define MPPIGENERIC_RACER_SUSPENSION_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/utils/angle_utils.cuh>

struct RacerSuspensionParams : public DynamicsParams
{
  enum class StateIndex : int
  {
    POS_X = 0,
    POS_Y,
    POS_Z,
    QUAT_W,
    QUAT_X,
    QUAT_Y,
    QUAT_Z,
    VX_I,
    VY_I,
    VZ_I,
    OMEGA_X,
    OMEGA_Y,
    OMEGA_Z,
    TRUE_STEER_ANGLE,
    TRUE_STEER_ANGLE_VEL,
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    BRAKE_THROTTLE = 0,
    DESIRED_STEERING,
    NUM_CONTROLS
  };
  // suspension model params
  float wheel_radius = 0.32;
  float mass = 1447;
  float wheel_base = 2.981;
  float width = 1.5;
  float height = 1.5;
  float gravity = 9.81;
  float k_s[4] = { 14000, 14000, 14000, 14000 };
  float c_s[4] = { 2000, 2000, 2000, 2000 };
  float l_0[4] = {
    wheel_radius + mass / 4 * gravity / k_s[0],
    wheel_radius + mass / 4 * gravity / k_s[1],
    wheel_radius + mass / 4 * gravity / k_s[2],
    wheel_radius + mass / 4 * gravity / k_s[3],
  };
  float3 cg_pos_wrt_base_link = make_float3(wheel_base / 2, 0, 0.5);
  // float3 wheel_pos_front_left = make_float3(wheel_base, width/2, 0);
  // float3 wheel_pos_front_right = make_float3(wheel_base, -width/2, 0);
  // float3 wheel_pos_rear_left = make_float3(0, width/2, 0);
  // float3 wheel_pos_rear_right = make_float3(0, -width/2, 0);
  float3 wheel_pos_wrt_base_link[4] = { make_float3(wheel_base, width / 2, 0), make_float3(wheel_base, -width / 2, 0),
                                        make_float3(0, width / 2, 0), make_float3(0, -width / 2, 0) };
  float Jxx = 1 / 12 * mass * height * height + width * width;
  float Jyy = 1 / 12 * mass * height * height + wheel_base * wheel_base;
  float Jzz = 1 / 12 * mass * wheel_base * wheel_base + width * width;
  float mu = 0.65;
  float v_slip = 0.1;
  static const int WHEEL_FRONT_LEFT = 0;
  static const int WHEEL_FRONT_RIGHT = 1;
  static const int WHEEL_REAR_LEFT = 2;
  static const int WHEEL_REAR_RIGHT = 3;

  // throttle model params
  float c_t = 1.3;
  float c_b = 2.5;
  float c_v = 3.7;
  float c_0 = 4.9;

  // steering model params
  float steering_constant = .6;
  float steer_command_angle_scale = -2.45;
};

using namespace MPPI_internal;
/**
 * state: x, y, z, quat [w, x, y, z], vx_I, vy_I, vz_I, omega_x, omega_y, omega_z, shaft angle, shaft angle velocity
 * control: throttle/brake, steering angle command
 * sensors: texture 0 is elevation map (normal x, normal y, normal z, height)
 */
class RacerSuspension : public Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>
{
public:
  // number of floats for computing the state derivative BLOCK_DIM_X * BLOCK_DIM_Z times
  static const int SHARED_MEM_REQUEST_BLK = 0;

  static const int STATE_P = 0;
  static const int STATE_PX = 0;
  static const int STATE_PY = 1;
  static const int STATE_PZ = 2;
  static const int STATE_Q = 3;
  static const int STATE_QW = 3;
  static const int STATE_QX = 4;
  static const int STATE_QY = 5;
  static const int STATE_QZ = 6;
  static const int STATE_V = 7;
  static const int STATE_VX = 7;
  static const int STATE_VY = 8;
  static const int STATE_VZ = 9;
  static const int STATE_OMEGA = 10;
  static const int STATE_OMEGAX = 10;
  static const int STATE_OMEGAY = 11;
  static const int STATE_OMEGAZ = 12;
  static const int STATE_STEER = 13;
  static const int STATE_STEER_VEL = 14;
  static const int CTRL_THROTTLE_BRAKE = 0;
  static const int CTRL_STEER_CMD = 1;
  static const int TEXTURE_ELEVATION_MAP = 0;

  RacerSuspension(cudaStream_t stream = nullptr) : Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>(stream)
  {
    tex_helper_ = new TwoDTextureHelper<float4>(1, stream);
  }
  RacerSuspension(RacerSuspensionParams& params, cudaStream_t stream = nullptr)
    : Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>(params, stream)
  {
    tex_helper_ = new TwoDTextureHelper<float4>(1, stream);
  }

  ~RacerSuspension()
  {
    delete tex_helper_;
  }

  void GPUSetup();

  void freeCudaMem();

  void paramsToDevice();

  void updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt);

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  __device__ void updateState(float* state, float* state_der, const float dt);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);

  TwoDTextureHelper<float4>* getTextureHelper()
  {
    return tex_helper_;
  }

  Eigen::Quaternionf get_attitude(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f get_position(const Eigen::Ref<const state_array>& state);

protected:
  TwoDTextureHelper<float4>* tex_helper_ = nullptr;
};

#if __CUDACC__
#include "racer_suspension.cu"
#endif

#endif  // MPPIGENERIC_RACER_SUSPENSION_CUH
