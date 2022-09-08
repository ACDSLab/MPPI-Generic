//
// Created by jason on 9/7/22.
//

#ifndef MPPIGENERIC_ACKERMAN_SLIP_CUH
#define MPPIGENERIC_ACKERMAN_SLIP_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/angle_utils.cuh>
#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>
#include "mppi/utils/texture_helpers/two_d_texture_helper.cuh"

struct AckermanSlipParams : public DynamicsParams
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
    ACCEL_Y,
    NUM_OUTPUTS
  };
  float wheel_angle_scale = -9.2;
  float gravity = -9.81;
  // steering parametric
  float steer_command_angle_scale = 5;
  float steering_constant = .6;
  float max_steer_angle = 5;
  float max_steer_rate = 5;
  // brake parametric component
  float brake_delay_constant = 6.6;
  float max_brake_rate_neg = 0.9;
  float max_brake_rate_pos = 0.33;
  // forward reverse
  int gear_sign = 1;
};


class AckermanSlip : public MPPI_internal::Dynamics<AckermanSlip, AckermanSlipParams>
{
public:
  using PARENT_CLASS = MPPI_internal::Dynamics<AckermanSlip, AckermanSlipParams>;
  typedef LSTMHelper<LSTMParams<6, 5>, FNNParams<11,20,1>> STEER_LSTM;
  typedef LSTMHelper<LSTMParams<5, 60>, FNNParams<65, 100, 10>> STEER_INIT_LSTM;
  typedef LSTMLSTMHelper<STEER_INIT_LSTM, STEER_LSTM, 51> STEER_NN;

  typedef LSTMHelper<LSTMParams<3, 5>, FNNParams<8,30,1>> DELAY_LSTM;
  typedef LSTMHelper<LSTMParams<1, 60>, FNNParams<61, 100, 10>> DELAY_INIT_LSTM;
  typedef LSTMLSTMHelper<DELAY_INIT_LSTM, DELAY_LSTM, 51> DELAY_NN;

  typedef LSTMHelper<LSTMParams<8, 15>, FNNParams<23,30,3>> TERRA_LSTM;
  typedef LSTMHelper<LSTMParams<7, 60>, FNNParams<67, 100, 30>> TERRA_INIT_LSTM;
  typedef LSTMLSTMHelper<TERRA_INIT_LSTM, TERRA_LSTM, 51> TERRA_NN;

  typedef LSTMHelper<LSTMParams<3, 5>, FNNParams<8, 20, 1>> ENGINE_LSTM;
  typedef LSTMHelper<LSTMParams<3, 60>, FNNParams<63, 100, 10>> ENGINE_INIT_LSTM;
  typedef LSTMLSTMHelper<ENGINE_INIT_LSTM, ENGINE_LSTM, 51> ENGINE_NN;

  struct SHARED_MEM_GRD_PARAMS {
    LSTMParams<6, 5> steer_lstm_params;
    FNNParams<11, 20, 1> steer_output_params;

    LSTMParams<3, 5> delay_lstm_params;
    FNNParams<8, 30, 1> delay_output_params;

    LSTMParams<8, 15> terra_lstm_params;
    FNNParams<23, 30, 3> terra_output_params;

    LSTMParams<3, 5> engine_lstm_params;
    FNNParams<8, 20, 1> engine_output_params;
  };

  struct SHARED_MEM_BLK_PARAMS {
    float steer_hidden_cell[2 * STEER_LSTM::HIDDEN_DIM];
    float delay_hidden_cell[2 * DELAY_LSTM::HIDDEN_DIM];
    float terra_hidden_cell[2 * TERRA_LSTM::HIDDEN_DIM];
    float engine_hidden_cell[2 * ENGINE_LSTM::HIDDEN_DIM];

    // terra is the largest, should be init'd smarter though
    float theta_s[TERRA_LSTM::HIDDEN_DIM * 4 + TERRA_LSTM::INPUT_DIM + TERRA_LSTM::OUTPUT_FNN_T::SHARED_MEM_REQUEST_BLK];
  };

  static const int SHARED_MEM_REQUEST_GRD = STEER_NN::SHARED_MEM_REQUEST_GRD
                                            + DELAY_NN::SHARED_MEM_REQUEST_GRD
                                            + TERRA_NN::SHARED_MEM_REQUEST_GRD
                                            + ENGINE_NN::SHARED_MEM_REQUEST_GRD;

  // TODO fix to use maximum and then assume no parallel computation
  static const int SHARED_MEM_REQUEST_BLK = sizeof(SHARED_MEM_BLK_PARAMS) / sizeof(float) + 1;

  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::output_array output_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;

  explicit AckermanSlip(cudaStream_t stream = nullptr);
  explicit AckermanSlip(std::string steering_path, std::string ackerman_path, cudaStream_t stream = nullptr);

  void paramsToDevice();

  void GPUSetup();

  void freeCudaMem();

  void updateFromBuffer(const buffer_trajectory& buffer);

  void initializeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                          Eigen::Ref<output_array> output, float t_0, float dt);

  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt);

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt);

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt);

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt);
  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);
  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);

  // state_array interpolateState(const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2,
  //                              const float alpha);

  // bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
  //                  Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);


  // __device__ void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s, float
  // *output=nullptr); // TODO

  // void getStoppingControl(const Eigen::Ref<const state_array>& state, Eigen::Ref<control_array> u);

  // Eigen::Quaternionf attitudeFromState(const Eigen::Ref<const state_array>& state);
  // Eigen::Vector3f positionFromState(const Eigen::Ref<const state_array>& state);
  // Eigen::Vector3f velocityFromState(const Eigen::Ref<const state_array>& state);
  // Eigen::Vector3f angularRateFromState(const Eigen::Ref<const state_array>& state);
  // state_array stateFromOdometry(const Eigen::Quaternionf& q, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel,
  //                               const Eigen::Vector3f& omega);

  // void enforceLeash(const Eigen::Ref<const state_array>& state_true, const Eigen::Ref<const state_array>& state_nominal,
  //                   const Eigen::Ref<const state_array>& leash_values, Eigen::Ref<state_array> state_output);

  state_array stateFromMap(const std::map<std::string, float>& map) override;

  std::shared_ptr<STEER_NN> getSteerHelper() {
    return steer_lstm_lstm_helper_;
  }
  std::shared_ptr<DELAY_NN> getDelayHelper() {
    return delay_lstm_lstm_helper_;
  }
  std::shared_ptr<ENGINE_NN> getEngineHelper() {
    return engine_lstm_lstm_helper_;
  }
  std::shared_ptr<TERRA_NN> getTerraHelper() {
    return terra_lstm_lstm_helper_;
  }
  TwoDTextureHelper<float>* getTextureHelper()
  {
    return tex_helper_;
  }

  STEER_LSTM* steer_network_d_ = nullptr;
  DELAY_LSTM* delay_network_d_ = nullptr;
  ENGINE_LSTM* engine_network_d_ = nullptr;
  TERRA_LSTM* terra_network_d_ = nullptr;
protected:
  TwoDTextureHelper<float>* tex_helper_ = nullptr;

  std::shared_ptr<STEER_NN> steer_lstm_lstm_helper_;
  std::shared_ptr<DELAY_NN> delay_lstm_lstm_helper_;
  std::shared_ptr<ENGINE_NN> engine_lstm_lstm_helper_;
  std::shared_ptr<TERRA_NN> terra_lstm_lstm_helper_;
};

#if __CUDACC__
#include "ackerman_slip.cu"
#endif

#endif  // MPPIGENERIC_ACKERMAN_SLIP_CUH
