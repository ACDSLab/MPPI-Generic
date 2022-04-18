#ifndef MPPIGENERIC_RACER_SUSPENSION_CUH
#define MPPIGENERIC_RACER_SUSPENSION_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/utils/angle_utils.cuh>

struct RacerSuspensionParams
{
  // suspension model params
  float wheel_radius = 0.32;
  float mass = 1447;
  float wheel_base = 3.0;
  float width = 1.5;
  float height = 1.5;
  float gravity = 0.0;
  float k_s = 14000;
  float c_s = 2000;
  float l_0 = (wheel_radius + mass/4 * gravity/k_s);
  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 back_left = make_float3(0, 0.737, 0);
  float3 back_right = make_float3(0, -0.737, 0);

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
 * state: x, y, z, quat, vx_I, vy_I, vz_I, omega_x, omega_y, omega_z, shaft angle, shaft angle velocity
 * control: throttle/brake, steering angle command
 * sensors: texture is (normal x, normal y, normal z, height)
 */
class RacerSuspension : public Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>
{
public:
  // number of floats for computing the state derivative BLOCK_DIM_X * BLOCK_DIM_Z times
  static const int SHARED_MEM_REQUEST_BLK = 0;

  RacerSuspension(cudaStream_t stream = nullptr) : Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>(stream)
  {
    tex_helper_ = new TwoDTextureHelper<float4>(1, stream);
  }
  RacerSuspension(RacerSuspensionParams& params, cudaStream_t stream = nullptr)
    : Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>(params, stream)
  {
    tex_helper_ = new TwoDTextureHelper<float4>(1, stream);
  }

  ~RacerSuspension() {
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

  TwoDTextureHelper<float4>* getTextureHelper() {return tex_helper_;}

protected:
  TwoDTextureHelper<float4>* tex_helper_ = nullptr;
};

#if __CUDACC__
#include "racer_suspension.cu"
#endif

#endif  // MPPIGENERIC_RACER_SUSPENSION_CUH
