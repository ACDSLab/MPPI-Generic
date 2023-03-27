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
    BRAKE_STATE,
    ROLL,
    PITCH,
    STEER_ANGLE_RATE,
    NUM_STATES
  };
};

template<class CLASS_T, class PARAMS_T>
class RacerDubinsElevationImpl : public RacerDubinsImpl<CLASS_T, PARAMS_T>
{
public:
  // static const int SHARED_MEM_REQUEST_GRD_BYTES = sizeof(DYN_PARAMS_T);
  using PARENT_CLASS = RacerDubinsImpl<CLASS_T, PARAMS_T>;
  using PARENT_CLASS::initializeDynamics;

  typedef PARAMS_T DYN_PARAMS_T;

  static const int SHARED_MEM_REQUEST_GRD_BYTES = 0;  // TODO set to one to prevent array of size 0 error
  static const int SHARED_MEM_REQUEST_BLK_BYTES = 0;

  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::output_array output_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;

  RacerDubinsElevationImpl(cudaStream_t stream = nullptr) : PARENT_CLASS(stream)
  {
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  }
  RacerDubinsElevationImpl(RacerDubinsElevationParams& params, cudaStream_t stream = nullptr) : PARENT_CLASS(params, stream)
  {
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  }

  std::string getDynamicsModelName() const override
  {
    return "RACER Dubins w/ Elevation Model";
  }

  ~RacerDubinsElevationImpl()
  {
    delete tex_helper_;
  }

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt) {
    this->PARENT_CLASS::updateState(state, next_state, state_der, dt);
  }

  void GPUSetup();

  void freeCudaMem();

  void paramsToDevice();

  void computeParametricAccelDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                              Eigen::Ref<state_array> state_der, const float dt);

  void computeParametricDelayDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                                   Eigen::Ref<state_array> state_der);

  void computeParametricSteerDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                                   Eigen::Ref<state_array> state_der);

  __device__ void computeParametricAccelDeriv(float* state, float* control,
                                   float* state_der, const float dt, DYN_PARAMS_T* params_p);

  __device__ void computeParametricDelayDeriv(float* state, float* control,
                                              float* state_der, DYN_PARAMS_T* params_p);

  __device__ void computeParametricSteerDeriv(float* state, float* control,
                                              float* state_der, DYN_PARAMS_T* params_p);

  __host__ __device__ void setOutputs(const float* state_der,
                  const float* next_state,
                  float* output);

  bool computeGrad(const Eigen::Ref<const state_array>& state = state_array(),
                   const Eigen::Ref<const control_array>& control = control_array(), Eigen::Ref<dfdx> A = dfdx(),
                   Eigen::Ref<dfdu> B = dfdu());


  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt, DYN_PARAMS_T* params_p);


  void computeStateDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                         Eigen::Ref<state_array> state_der)
  {
  }

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt);

  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);

  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt);

  Eigen::Quaternionf attitudeFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f positionFromState(const Eigen::Ref<const state_array>& state);

  state_array stateFromMap(const std::map<std::string, float>& map) override;

  TwoDTextureHelper<float>* getTextureHelper()
  {
    return tex_helper_;
  }

protected:
  TwoDTextureHelper<float>* tex_helper_ = nullptr;
};

class RacerDubinsElevation : public RacerDubinsElevationImpl<RacerDubinsElevation, RacerDubinsElevationParams>
{
public:
  RacerDubinsElevation(cudaStream_t stream=nullptr) : RacerDubinsElevationImpl<RacerDubinsElevation, RacerDubinsElevationParams>(stream) {}
  RacerDubinsElevation(RacerDubinsElevationParams& params, cudaStream_t stream=nullptr) : RacerDubinsElevationImpl<RacerDubinsElevation, RacerDubinsElevationParams>(params, stream) {}
};

#if __CUDACC__
#include "racer_dubins_elevation.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH
