//
// Created by jason on 8/31/22.
//

#include "racer_dubins_elevation.cuh"
#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>

#ifndef MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_STEERING_CUH
#define MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_STEERING_CUH

class RacerDubinsElevationLSTMSteering : public RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering, RacerDubinsElevationParams>
{
public:
  using PARENT_CLASS = RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering, RacerDubinsElevationParams>;
  typedef FNNParams<10, 20, 1> FNN_PARAMS;
  typedef FNNParams<64, 100, 10> FNN_INIT_PARAMS;
  typedef LSTMHelper<LSTMParams<5, 5>, FNN_PARAMS> LSTM;
  typedef LSTMHelper<LSTMParams<4, 60>, FNN_INIT_PARAMS> INIT_LSTM;
  typedef LSTMLSTMHelper<INIT_LSTM, LSTM, 51> NN;

  static const int SHARED_MEM_REQUEST_GRD_BYTES = RacerDubinsElevation::SHARED_MEM_REQUEST_GRD_BYTES +
      NN::SHARED_MEM_REQUEST_GRD_BYTES;
  static const int SHARED_MEM_REQUEST_BLK_BYTES = RacerDubinsElevation::SHARED_MEM_REQUEST_BLK_BYTES +
                                            NN::SHARED_MEM_REQUEST_BLK_BYTES;

  RacerDubinsElevationLSTMSteering(cudaStream_t stream = nullptr);
  RacerDubinsElevationLSTMSteering(RacerDubinsElevationParams& params, cudaStream_t stream = nullptr);
  RacerDubinsElevationLSTMSteering(std::string path, cudaStream_t stream = nullptr);

  std::string getDynamicsModelName() const override
  {
    return "RACER Dubins LSTM Steering Model";
  }

  void GPUSetup();

  void freeCudaMem();

  void updateFromBuffer(const buffer_trajectory& buffer);

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt);

  void initializeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                          Eigen::Ref<output_array> output, float t_0, float dt);

  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);

  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt);

  std::shared_ptr<NN> getHelper() {
    return lstm_lstm_helper_;
  }

  LSTM* network_d_ = nullptr;
protected:
  std::shared_ptr<NN> lstm_lstm_helper_;
};

const int RacerDubinsElevationLSTMSteering::SHARED_MEM_REQUEST_GRD_BYTES;
const int RacerDubinsElevationLSTMSteering::SHARED_MEM_REQUEST_BLK_BYTES;

#if __CUDACC__
#include "racer_dubins_elevation_lstm_steering.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_STEERING_CUH
