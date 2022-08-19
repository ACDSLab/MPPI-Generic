#ifndef MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH
#define MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/utils/angle_utils.cuh>

using namespace MPPI_internal;

struct RacerDubinsElevationParams : public RacerDubinsParams
{
  enum class StateIndex : int
  {
    VEL_X = 0,
    YAW,
    POS_X,
    POS_Y,
    STEER_ANGLE,
    ROLL,
    PITCH,
    STEER_ANGLE_RATE,
    ACCEL_X,
    NUM_STATES
  };
};

class RacerDubinsElevation : public RacerDubinsImpl<RacerDubinsElevation, RacerDubinsElevationParams>
{
public:
  // static const int SHARED_MEM_REQUEST_GRD = sizeof(DYN_PARAMS_T);
  using PARENT_CLASS = RacerDubinsImpl<RacerDubinsElevation, RacerDubinsElevationParams>;
  using PARENT_CLASS::initializeDynamics;
  RacerDubinsElevation(cudaStream_t stream = nullptr) : PARENT_CLASS(stream)
  {
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  }
  RacerDubinsElevation(RacerDubinsElevationParams& params, cudaStream_t stream = nullptr) : PARENT_CLASS(params, stream)
  {
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  }

  ~RacerDubinsElevation()
  {
    delete tex_helper_;
  }

  void GPUSetup();

  void freeCudaMem();

  void paramsToDevice();

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt);

  void computeStateDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                         Eigen::Ref<state_array> state_der)
  {
  }

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt);

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt)
  {
  }

  __device__ void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s)
  {
  }

  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);

  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt);

  TwoDTextureHelper<float>* getTextureHelper()
  {
    return tex_helper_;
  }

protected:
  TwoDTextureHelper<float>* tex_helper_ = nullptr;
};

#if __CUDACC__
#include "racer_dubins_elevation.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH
