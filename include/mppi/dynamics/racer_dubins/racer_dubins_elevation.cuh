#ifndef MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH
#define MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/utils/angle_utils.cuh>

using namespace MPPI_internal;

class RacerDubinsElevation : public RacerDubinsImpl<RacerDubinsElevation>
{
public:
  RacerDubinsElevation(cudaStream_t stream = nullptr) : RacerDubinsImpl<RacerDubinsElevation>(stream)
  {
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  }
  RacerDubinsElevation(RacerDubinsParams& params, cudaStream_t stream = nullptr)
    : RacerDubinsImpl<RacerDubinsElevation>(params, stream)
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

  void updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt);

  void computeStateDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                          Eigen::Ref<state_array> state_der, output_array* output=nullptr);

  __device__ void updateState(float* state, float* state_der, const float dt);

  __device__ void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s, float *output=nullptr);


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
