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

template<class CLASS_T>
class RacerDubinsElevationImpl : public RacerDubinsImpl<CLASS_T, RacerDubinsElevationParams>
{
public:
  // static const int SHARED_MEM_REQUEST_GRD = sizeof(DYN_PARAMS_T);
  using PARENT_CLASS = RacerDubinsImpl<CLASS_T, RacerDubinsElevationParams>;
  using PARENT_CLASS::initializeDynamics;

  typedef RacerDubinsElevationParams DYN_PARAMS_T;

  static const int SHARED_MEM_REQUEST_GRD = 1;  // TODO set to one to prevent array of size 0 error
  static const int SHARED_MEM_REQUEST_BLK = 0;

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

  void computeParametricModelDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                              Eigen::Ref<state_array> state_der, const float dt);

  __device__ void computeParametricModelDeriv(float* state, float* control,
                                   float* state_der, const float dt, DYN_PARAMS_T* params_p);

  __host__ __device__ void setOutputs(const float* state_der,
                  const float* next_state,
                  float* output);

  __device__ __host__ void computeStaticSettling(const float yaw, const float x, const float y,
                                                 float roll, float pitch, float* output);

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

  state_array stateFromMap(const std::map<std::string, float>& map) override;

  TwoDTextureHelper<float>* getTextureHelper()
  {
    return tex_helper_;
  }

protected:
  TwoDTextureHelper<float>* tex_helper_ = nullptr;
};

class RacerDubinsElevation : public RacerDubinsElevationImpl<RacerDubinsElevation>
{
public:
  RacerDubinsElevation(cudaStream_t stream=nullptr) : RacerDubinsElevationImpl<RacerDubinsElevation>(stream) {}
  RacerDubinsElevation(RacerDubinsElevationParams& params, cudaStream_t stream=nullptr) : RacerDubinsElevationImpl<RacerDubinsElevation>(params, stream) {}
};

#if __CUDACC__
#include "racer_dubins_elevation.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_CUH
