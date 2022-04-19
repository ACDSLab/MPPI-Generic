#ifndef MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH
#define MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/utils/angle_utils.cuh>

using namespace MPPI_internal;
/**
 * state: v, theta, p_x, p_y, true steering angle
 * control: throttle, steering angle command
 */
class RacerDubinsElevation : public RacerDubinsImpl<RacerDubinsElevation, 7>
{
public:
  RacerDubinsElevation(cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubinsElevation, 7>(stream)
  {
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  }
  RacerDubinsElevation(RacerDubinsParams& params, cudaStream_t stream = nullptr)
    : RacerDubinsImpl<RacerDubinsElevation, 7>(params, stream)
  {
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  }

  ~RacerDubinsElevation() {
    delete tex_helper_;
  }

  void GPUSetup();

  void freeCudaMem();

  void paramsToDevice();

  void updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt);

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der) {
    RacerDubinsImpl<RacerDubinsElevation, 7>* derived = static_cast<RacerDubinsImpl<RacerDubinsElevation, 7>*>(this);
    derived->computeDynamics(state, control, state_der);
  }

  __device__ void updateState(float* state, float* state_der, const float dt);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);

  TwoDTextureHelper<float>* getTextureHelper() {return tex_helper_;}

protected:
  TwoDTextureHelper<float>* tex_helper_ = nullptr;
};

#if __CUDACC__
#include "racer_dubins_elevation.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH
