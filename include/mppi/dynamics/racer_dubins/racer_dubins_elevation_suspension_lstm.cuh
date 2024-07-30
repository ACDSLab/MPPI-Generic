//
// Created by Bogdan on 02/21/2024
//
#pragma once

#include "racer_dubins_elevation_lstm_steering.cuh"

#ifndef W_INDEX
#define W_IND_CLASS(CLASS, enum_val) E_INDEX(CLASS::WheelIndex, enum_val)
#define W_IND(param, enum_val) W_IND_CLASS(decltype(param), enum_val)
#define W_INDEX(enum_val) W_IND(this->params_, enum_val)
#define W_INDEX_NEWLY_CREATED true
#endif

struct RacerDubinsElevationSuspensionParams : public RacerDubinsElevationParams
{
  enum class WheelIndex : int
  {
    FL = 0,
    FR,
    BL,
    BR,
    NUM_WHEELS,
  };

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
    CG_POS_Z,
    CG_VEL_I_Z,
    ROLL_RATE,
    PITCH_RATE,
    STEER_ANGLE_RATE,
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
    FILLER_1,
    NUM_STATES
  };

  float spring_k = 14000.0f;                              // [N / m]
  float drag_c = 1000.0f;                                 // [N * s / m]
  float mass = 1447.0f;                                   // [kg]
  float I_xx = 1.0f / 12 * mass * 2 * SQ(1.5f);           // [kg * m^2]
  float I_yy = 1.0f / 12 * mass * (SQ(1.5f) + SQ(3.0f));  // [kg * m^2]
  float wheel_radius = 0.32f;                             // [m]

  // Cost force threshold on the order of 3000 N
  // TODO Figure out Center of Gravity
  float3 c_g = make_float3(2.981f * 0.5f, 0.0f, 0.0f);
};

template <class CLASS_T, class PARAMS_T = RacerDubinsElevationSuspensionParams>
class RacerDubinsElevationSuspensionImpl : public RacerDubinsElevationLSTMSteeringImpl<CLASS_T, PARAMS_T>
{
public:
  using PARENT_CLASS = RacerDubinsElevationLSTMSteeringImpl<CLASS_T, PARAMS_T>;
  using NN = typename PARENT_CLASS::NN;
  using LSTM = typename PARENT_CLASS::LSTM;
  using GRANDPARENT_CLASS = typename PARENT_CLASS::PARENT_CLASS;
  using DYN_PARAMS_T = typename PARENT_CLASS::DYN_PARAMS_T;
  using SharedBlock = typename PARENT_CLASS::SharedBlock;
  using state_array = typename PARENT_CLASS::state_array;
  using control_array = typename PARENT_CLASS::control_array;
  using output_array = typename PARENT_CLASS::output_array;
  using buffer_trajectory = typename PARENT_CLASS::buffer_trajectory;
  using PARENT_CLASS::computeParametricAccelDeriv;
  using PARENT_CLASS::computeParametricDelayDeriv;
  using PARENT_CLASS::computeUncertaintyPropagation;

  RacerDubinsElevationSuspensionImpl(cudaStream_t stream = nullptr);
  RacerDubinsElevationSuspensionImpl(PARAMS_T& params, cudaStream_t stream = nullptr);
  RacerDubinsElevationSuspensionImpl(std::string path, cudaStream_t stream = nullptr);

  std::string getDynamicsModelName() const override
  {
    return "RACER Dubins LSTM Steering and Suspension Model";
  }

  void paramsToDevice();

  void GPUSetup();

  void freeCudaMem();

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt);

  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt);

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt);

  __host__ __device__ void setOutputs(const float* state_der, const float* next_state, float* output);

  TwoDTextureHelper<float4>* getTextureHelperNormals()
  {
    return normals_tex_helper_;
  }

  void updateRotation(std::array<float3, 3>& rotation)
  {
    this->tex_helper_->updateRotation(0, rotation);
    this->normals_tex_helper_->updateRotation(0, rotation);
  }

  state_array stateFromMap(const std::map<std::string, float>& map);

protected:
  TwoDTextureHelper<float4>* normals_tex_helper_ = nullptr;
};

class RacerDubinsElevationSuspension : public RacerDubinsElevationSuspensionImpl<RacerDubinsElevationSuspension>
{
public:
  using PARENT_CLASS = RacerDubinsElevationSuspensionImpl<RacerDubinsElevationSuspension>;

  RacerDubinsElevationSuspension(cudaStream_t stream = 0) : PARENT_CLASS(stream)
  {
  }
  RacerDubinsElevationSuspension(RacerDubinsElevationSuspensionParams params, cudaStream_t stream = 0)
    : PARENT_CLASS(params, stream)
  {
  }
  RacerDubinsElevationSuspension(std::string path, cudaStream_t stream = 0) : PARENT_CLASS(path, stream = 0)
  {
  }
};

#ifdef __CUDACC__
#include "racer_dubins_elevation_suspension_lstm.cu"
#endif

#if W_INDEX_NEWLY_CREATED
#undef W_IND_CLASS
#undef W_IND
#undef W_INDEX
#endif
