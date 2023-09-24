//
// Created by jason on 12/12/22.
//

#ifndef MPPIGENERIC_BICYCLE_SLIP_PARAMTERIC_CUH
#define MPPIGENERIC_BICYCLE_SLIP_PARAMTERIC_CUH

#include <mppi/utils/angle_utils.cuh>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/utils/activation_functions.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>

#define BICYCLE_UNCERTAINTY

namespace RACER
{
template <class TEX_T>
__device__ __host__ void computeBodyFrameNormals(TEX_T* tex_helper, const float yaw, const float x, const float y,
                                                 const float roll, const float pitch, float& mean_normals_x,
                                                 float& mean_normals_y, float& mean_normals_z);
};

struct BicycleSlipParametricParams : public RacerDubinsElevationParams
{
  enum class StateIndex : int
  {
    POS_X = 0,
    POS_Y,
    YAW,
    STEER_ANGLE,
    BRAKE_STATE,
    VEL_X,
    VEL_Y,
    OMEGA_Z,
    ROLL,
    PITCH,
    STEER_ANGLE_RATE,
    FILLER_1, // TODO: Figure out if more filler is necessary
    UNCERTAINTY_POS_X,
    UNCERTAINTY_POS_Y,
    UNCERTAINTY_YAW,
    UNCERTAINTY_VEL_X,
    UNCERTAINTY_POS_X_Y,
    UNCERTAINTY_POS_X_YAW,
    UNCERTAINTY_POS_X_VEL_X,
    UNCERTAINTY_POS_Y_YAW,
    UNCERTAINTY_POS_Y_VEL_X,
    UNCERTAINTY_YAW_VEL_X,
    NUM_STATES
  };

  float environment = 1.0f;  // 1.0 for helendale, -1.0 for halter ranch

  float mu = 4.7f;
  float mu_env = 2.0f;
  float gravity_x = -3.9f;
  float min_normal_x = 0.155f;
  float gravity_y = -7.2f;
  float min_normal_y = 0.24f;
  float brake_vel = 0.9f;
  float c_vy = 6.5f;
  float vy_omega = 1.6f;
  float c_rolling = 0.2f;
  float max_roll_resistance_vel = 0.8f;
  float c_sliding = 0.2f;
  float max_slide_vel = 0.8f;

  float c_omega = 2.2f;
  float c_v_omega = 4.2f;

  BicycleSlipParametricParams()
  {
    c_t[0] = 2.73f;
    c_t[1] = 0.15f;
    c_t[2] = -0.0145f;

    c_b[0] = 30.0f;
    c_v[0] = 0.079f;  // c_vx
  }
};

template <class CLASS_T, class PARAMS_T>
class BicycleSlipParametricImpl : public RacerDubinsElevationImpl<CLASS_T, PARAMS_T>
{
public:
  using PARENT_CLASS = RacerDubinsElevationImpl<CLASS_T, PARAMS_T>;

  typedef PARAMS_T DYN_PARAMS_T;

  static const int SHARED_MEM_REQUEST_GRD_BYTES = sizeof(PARAMS_T);  // TODO set to one to prevent array of size 0 error
  static const int SHARED_MEM_REQUEST_BLK_BYTES = PARENT_CLASS::SHARED_MEM_REQUEST_BLK_BYTES;

  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::output_array output_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;

  BicycleSlipParametricImpl(cudaStream_t stream = nullptr);
  BicycleSlipParametricImpl(const std::string& model_path, cudaStream_t stream = nullptr);

  std::string getDynamicsModelName() const override
  {
    return "Bicycle Slip Parametric Model";
  }

  void GPUSetup();

  void freeCudaMem();

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt);

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt);

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt)
  {
  }
  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt,
                              typename PARENT_CLASS::DYN_PARAMS_T* params_p);
  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);
  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);
#ifdef BICYCLE_UNCERTAINTY
  __host__ __device__ bool computeUncertaintyJacobian(const float* state, const float* control, float* A,
                                                      PARAMS_T* params_p);
#endif

  void paramsToDevice();

  // state_array interpolateState(const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2,
  //                              const float alpha);

  // bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
  //                  Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);

  Eigen::Vector3f velocityFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f angularRateFromState(const Eigen::Ref<const state_array>& state);

  // void enforceLeash(const Eigen::Ref<const state_array>& state_true, const Eigen::Ref<const state_array>&
  // state_nominal,
  //                   const Eigen::Ref<const state_array>& leash_values, Eigen::Ref<state_array> state_output);

  state_array stateFromMap(const std::map<std::string, float>& map) override;

  TwoDTextureHelper<float4>* getTextureHelperNormals()
  {
    return normals_tex_helper_;
  }

protected:
  TwoDTextureHelper<float4>* normals_tex_helper_ = nullptr;
};

class BicycleSlipParametric : public BicycleSlipParametricImpl<BicycleSlipParametric, BicycleSlipParametricParams>
{
public:
  BicycleSlipParametric(cudaStream_t stream = nullptr)
    : BicycleSlipParametricImpl<BicycleSlipParametric, BicycleSlipParametricParams>(stream)
  {
  }
  BicycleSlipParametric(const std::string& model_path, cudaStream_t stream = nullptr)
    : BicycleSlipParametricImpl<BicycleSlipParametric, BicycleSlipParametricParams>(model_path, stream)
  {
  }
};

#if __CUDACC__
#include "bicycle_slip_parametric.cu"
#endif

#endif  // MPPIGENERIC_BICYCLE_SLIP_PARAMTERIC_CUH
