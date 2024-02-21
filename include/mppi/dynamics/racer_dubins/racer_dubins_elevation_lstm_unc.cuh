#include "racer_dubins_elevation_lstm_steering.cuh"

#ifndef MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_UNC_CUH
#define MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_UNC_CUH

class RacerDubinsElevationLSTMUncertainty
  : public RacerDubinsElevationImpl<RacerDubinsElevationLSTMUncertainty, RacerDubinsElevationParams>
{
public:
  using PARENT_CLASS = RacerDubinsElevationImpl<RacerDubinsElevationLSTMUncertainty, RacerDubinsElevationParams>;

  typedef typename PARENT_CLASS::DYN_PARAMS_T DYN_PARAMS_T;

  // struct SHARED_MEM_BLK_PARAMS
  // {
  //   float steer_hidden_cell[2 * STEER_LSTM::HIDDEN_DIM];
  //   float delay_hidden_cell[2 * UNC_LSTM::HIDDEN_DIM];
  //   float terra_hidden_cell[2 * MEAN_LSTM::HIDDEN_DIM];

  //   // terra is the largest, should be init'd smarter though
  //   float theta_s[UNC_LSTM::HIDDEN_DIM + UNC_LSTM::INPUT_DIM +
  //                 UNC_LSTM::OUTPUT_FNN_T::SHARED_MEM_REQUEST_BLK_BYTES / sizeof(float) + 1];
  // };

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

  std::shared_ptr<LSTMLSTMHelper<>> getSteerHelper()
  {
    return steer_helper_;
  }
  std::shared_ptr<LSTMLSTMHelper<>> getUncertaintyHelper()
  {
    return uncertainty_helper_;
  }
  std::shared_ptr<LSTMLSTMHelper<>> getMeanHelper()
  {
    return mean_helper_;
  }

  LSTMHelper<>* steer_d_ = nullptr;
  LSTMHelper<>* uncertainty_d_ = nullptr;
  LSTMHelper<>* mean_d_ = nullptr;

protected:
  std::shared_ptr<LSTMLSTMHelper<>> steer_helper_;
  std::shared_ptr<LSTMLSTMHelper<>> uncertainty_helper_;
  std::shared_ptr<LSTMLSTMHelper<>> mean_helper_;
};

// const int RacerDubinsElevationLSTMUncertainty::SHARED_MEM_REQUEST_GRD_BYTES;
// const int RacerDubinsElevationLSTMUncertainty::SHARED_MEM_REQUEST_BLK_BYTES;

#if __CUDACC__
#include "racer_dubins_elevation_lstm_unc.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_UNC_CUH
