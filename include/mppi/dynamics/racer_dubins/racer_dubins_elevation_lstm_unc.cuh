#include "racer_dubins_elevation_lstm_steering.cuh"

#ifndef MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_UNC_CUH
#define MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_UNC_CUH

class RacerDubinsElevationLSTMUncertainty
  : public RacerDubinsElevationImpl<RacerDubinsElevationLSTMUncertainty, RacerDubinsElevationParams>
{
public:
  using PARENT_CLASS = RacerDubinsElevationImpl<RacerDubinsElevationLSTMUncertainty, RacerDubinsElevationParams>;
  typedef LSTMHelper<LSTMParams<4, 4>, FNNParams<10, 20, 1>, false> STEER_LSTM;
  typedef LSTMHelper<LSTMParams<3, 20>, FNNParams<43, 100, 8>> STEER_INIT_LSTM;
  typedef LSTMLSTMHelper<STEER_INIT_LSTM, STEER_LSTM, 11> STEER_NN;

  typedef LSTMHelper<LSTMParams<10, 4>, FNNParams<14, 20, 5>, false> UNC_LSTM;
  typedef LSTMHelper<LSTMParams<10, 20>, FNNParams<50, 100, 8>> UNC_INIT_LSTM;
  typedef LSTMLSTMHelper<UNC_INIT_LSTM, UNC_LSTM, 11> UNC_NN;

  typedef LSTMHelper<LSTMParams<10, 4>, FNNParams<14, 20, 2>, false> MEAN_LSTM;
  typedef LSTMHelper<LSTMParams<10, 20>, FNNParams<50, 100, 8>> MEAN_INIT_LSTM;
  typedef LSTMLSTMHelper<MEAN_INIT_LSTM, MEAN_LSTM, 11> MEAN_NN;

  typedef typename PARENT_CLASS::DYN_PARAMS_T DYN_PARAMS_T;

  struct SHARED_MEM_GRD_PARAMS
  {
    LSTMParams<4, 4> steer_lstm_params;
    FNNParams<10, 20, 1> steer_output_params;

    LSTMParams<10, 4> unc_lstm_params;
    FNNParams<14, 20, 5> unc_output_params;

    LSTMParams<10, 4> mean_lstm_params;
    FNNParams<14, 20, 2> mean_output_params;
  };

  struct SHARED_MEM_BLK_PARAMS
  {
    float steer_hidden_cell[2 * STEER_LSTM::HIDDEN_DIM];
    float delay_hidden_cell[2 * UNC_LSTM::HIDDEN_DIM];
    float terra_hidden_cell[2 * MEAN_LSTM::HIDDEN_DIM];

    // terra is the largest, should be init'd smarter though
    float theta_s[UNC_LSTM::HIDDEN_DIM + UNC_LSTM::INPUT_DIM +
                  UNC_LSTM::OUTPUT_FNN_T::SHARED_MEM_REQUEST_BLK_BYTES / sizeof(float) + 1];
  };

  static const int SHARED_MEM_REQUEST_GRD_BYTES = STEER_NN::SHARED_MEM_REQUEST_GRD_BYTES +
                                                  UNC_NN::SHARED_MEM_REQUEST_GRD_BYTES +
                                                  MEAN_NN::SHARED_MEM_REQUEST_GRD_BYTES;

  // TODO fix to use maximum and then assume no parallel computation
  static const int SHARED_MEM_REQUEST_BLK_BYTES = sizeof(SHARED_MEM_BLK_PARAMS);

  typedef typename PARENT_CLASS::state_array state_array;
  typedef typename PARENT_CLASS::control_array control_array;
  typedef typename PARENT_CLASS::output_array output_array;
  typedef typename PARENT_CLASS::dfdx dfdx;
  typedef typename PARENT_CLASS::dfdu dfdu;

  RacerDubinsElevationLSTMUncertainty(cudaStream_t stream = nullptr);
  RacerDubinsElevationLSTMUncertainty(RacerDubinsElevationParams& params, cudaStream_t stream = nullptr);
  RacerDubinsElevationLSTMUncertainty(std::string path, cudaStream_t stream = nullptr);

  std::string getDynamicsModelName() const override
  {
    return "RACER Dubins LSTM Uncertainty Model";
  }

  // TODO needs a rewrite of the compute Q function to run the NN
  // TODO need to change updateState for the accel version

  // void GPUSetup();

  // void freeCudaMem();

  // void updateFromBuffer(const buffer_trajectory& buffer);

  // void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
  //           const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
  //           const float dt);

  // void initializeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
  //                         Eigen::Ref<output_array> output, float t_0, float dt);

  // __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
  //                             float* theta_s, const float t, const float dt);

  // __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float
  // dt);

  // __device__ void updateState(float* state, float* next_state, float* state_der, const float dt);

  // void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
  //                  Eigen::Ref<state_array> state_der, const float dt);

  std::shared_ptr<STEER_NN> getSteerHelper()
  {
    return steer_helper_;
  }
  std::shared_ptr<UNC_NN> getUncertaintyHelper()
  {
    return uncertainty_helper_;
  }
  std::shared_ptr<MEAN_NN> getMeanHelper()
  {
    return mean_helper_;
  }

  STEER_LSTM* steer_d_ = nullptr;
  UNC_LSTM* uncertainty_d_ = nullptr;
  MEAN_LSTM* mean_d_ = nullptr;

protected:
  std::shared_ptr<STEER_NN> steer_helper_;
  std::shared_ptr<UNC_NN> uncertainty_helper_;
  std::shared_ptr<MEAN_NN> mean_helper_;
};

// const int RacerDubinsElevationLSTMUncertainty::SHARED_MEM_REQUEST_GRD_BYTES;
// const int RacerDubinsElevationLSTMUncertainty::SHARED_MEM_REQUEST_BLK_BYTES;

#if __CUDACC__
#include "racer_dubins_elevation_lstm_unc.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_UNC_CUH
