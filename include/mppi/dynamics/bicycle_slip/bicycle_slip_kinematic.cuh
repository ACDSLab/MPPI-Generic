//
// Created by jason on 12/12/22.
//

#ifndef MPPIGENERIC_BICYCLE_SLIP_KINEMATIC_CUH
#define MPPIGENERIC_BICYCLE_SLIP_KINEMATIC_CUH

#include <mppi/utils/angle_utils.cuh>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>

struct BicycleSlipKinematicParams : public RacerDubinsParams
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
    FILLER_1,
    NUM_STATES
  };

  float environment = 1.0f; // 1.0 for helendale, -1.0 for halter ranch
  bool enable_delay_model = true;
};

template<class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM = 9>
class BicycleSlipKinematicImpl : public RacerDubinsElevationImpl<CLASS_T, PARAMS_T>
{
 public:
  using PARENT_CLASS = RacerDubinsElevationImpl<CLASS_T, PARAMS_T>;
  typedef LSTMHelper<LSTMParams<5, 4>, FNNParams<9,5,1>, false> STEER_LSTM;
  typedef LSTMHelper<LSTMParams<4, 20>, FNNParams<24, 100, 8>> STEER_INIT_LSTM;
  typedef LSTMLSTMHelper<STEER_INIT_LSTM, STEER_LSTM, 51> STEER_NN;

  typedef LSTMHelper<LSTMParams<3, 4>, FNNParams<7,1>, false> DELAY_LSTM;
  typedef LSTMHelper<LSTMParams<2, 20>, FNNParams<22, 100, 8>> DELAY_INIT_LSTM;
  typedef LSTMLSTMHelper<DELAY_INIT_LSTM, DELAY_LSTM, 51> DELAY_NN;

  typedef LSTMHelper<LSTMParams<TERRA_INPUT_DIM, 20>, FNNParams<20 + TERRA_INPUT_DIM,20,3>, false> TERRA_LSTM;
  typedef LSTMHelper<LSTMParams<9, 40>, FNNParams<49, 400, 40>> TERRA_INIT_LSTM;
  typedef LSTMLSTMHelper<TERRA_INIT_LSTM, TERRA_LSTM, 51> TERRA_NN;

  typedef typename PARENT_CLASS::DYN_PARAMS_T DYN_PARAMS_T;

  struct SHARED_MEM_GRD_PARAMS {
    LSTMParams<5, 4> steer_lstm_params;
    FNNParams<9, 5, 1> steer_output_params;

    LSTMParams<3, 4> delay_lstm_params;
    FNNParams<7, 1> delay_output_params;

    LSTMParams<TERRA_INPUT_DIM, 20> terra_lstm_params;
    FNNParams<20 + TERRA_INPUT_DIM, 20, 3> terra_output_params;
  };

  struct SHARED_MEM_BLK_PARAMS {
    float steer_hidden_cell[2 * STEER_LSTM::HIDDEN_DIM];
    float delay_hidden_cell[2 * DELAY_LSTM::HIDDEN_DIM];
    float terra_hidden_cell[2 * TERRA_LSTM::HIDDEN_DIM];

    // terra is the largest, should be init'd smarter though
    float theta_s[TERRA_LSTM::HIDDEN_DIM + TERRA_LSTM::INPUT_DIM + TERRA_LSTM::OUTPUT_FNN_T::SHARED_MEM_REQUEST_BLK];
  };

  static const int SHARED_MEM_REQUEST_GRD = STEER_NN::SHARED_MEM_REQUEST_GRD
      + DELAY_NN::SHARED_MEM_REQUEST_GRD
      + TERRA_NN::SHARED_MEM_REQUEST_GRD;

  // TODO fix to use maximum and then assume no parallel computation
  static const int SHARED_MEM_REQUEST_BLK = sizeof(SHARED_MEM_BLK_PARAMS) / sizeof(float) + 1;

  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::output_array output_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;

  BicycleSlipKinematicImpl(cudaStream_t stream = nullptr);
  BicycleSlipKinematicImpl(const std::string& model_path, cudaStream_t stream = nullptr);

  std::string getDynamicsModelName() const override
  {
    return "Bicycle Slip Kinematic Model";
  }

  void GPUSetup();

  void freeCudaMem();

  void updateFromBuffer(const typename PARENT_CLASS::buffer_trajectory& buffer);

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

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt) {}
  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt, typename PARENT_CLASS::DYN_PARAMS_T* params_p);
  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);
  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);

  // state_array interpolateState(const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2,
  //                              const float alpha);

  // bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
  //                  Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);


  Eigen::Vector3f velocityFromState(const Eigen::Ref<const state_array>& state);
  Eigen::Vector3f angularRateFromState(const Eigen::Ref<const state_array>& state);

  // void enforceLeash(const Eigen::Ref<const state_array>& state_true, const Eigen::Ref<const state_array>& state_nominal,
  //                   const Eigen::Ref<const state_array>& leash_values, Eigen::Ref<state_array> state_output);

  state_array stateFromMap(const std::map<std::string, float>& map) override;

  std::shared_ptr<STEER_NN> getSteerHelper() {
    return steer_lstm_lstm_helper_;
  }
  std::shared_ptr<DELAY_NN> getDelayHelper() {
    return delay_lstm_lstm_helper_;
  }
  std::shared_ptr<TERRA_NN> getTerraHelper() {
    return terra_lstm_lstm_helper_;
  }

  STEER_LSTM* steer_network_d_ = nullptr;
  DELAY_LSTM* delay_network_d_ = nullptr;
  TERRA_LSTM* terra_network_d_ = nullptr;

 protected:
  std::shared_ptr<STEER_NN> steer_lstm_lstm_helper_;
  std::shared_ptr<DELAY_NN> delay_lstm_lstm_helper_;
  std::shared_ptr<TERRA_NN> terra_lstm_lstm_helper_;

};

class BicycleSlipKinematic : public BicycleSlipKinematicImpl<BicycleSlipKinematic, BicycleSlipKinematicParams>
{
 public:
  BicycleSlipKinematic(cudaStream_t stream = nullptr) : BicycleSlipKinematicImpl<BicycleSlipKinematic, BicycleSlipKinematicParams>(stream){}
  BicycleSlipKinematic(const std::string& model_path, cudaStream_t stream = nullptr) : BicycleSlipKinematicImpl<
      BicycleSlipKinematic, BicycleSlipKinematicParams>(model_path, stream){}
};

#if __CUDACC__
#include "bicycle_slip_kinematic.cu"
#endif

#endif  // MPPIGENERIC_BICYCLE_SLIP_KINEMATIC_CUH
