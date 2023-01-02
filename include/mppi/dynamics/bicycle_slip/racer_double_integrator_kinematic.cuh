//
// Created by jason on 12/12/22.
//

#ifndef MPPIGENERIC_DOUBLE_INTEGRATOR_KINEMATIC_CUH
#define MPPIGENERIC_DOUBLE_INTEGRATOR_KINEMATIC_CUH

#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/utils/angle_utils.cuh>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>
#include "mppi/utils/texture_helpers/two_d_texture_helper.cuh"

struct RacerDoubleIntegratorKinematicParams : public RacerDubinsParams
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
};

class RacerDoubleIntegratorKinematic : public MPPI_internal::Dynamics<RacerDoubleIntegratorKinematic, RacerDoubleIntegratorKinematicParams>
{
public:
    using PARENT_CLASS = MPPI_internal::Dynamics<RacerDoubleIntegratorKinematic, RacerDoubleIntegratorKinematicParams>;
    typedef LSTMHelper<LSTMParams<5, 5>, FNNParams<10,20,1>, false> STEER_LSTM;
    typedef LSTMHelper<LSTMParams<4, 60>, FNNParams<64, 100, 10>> STEER_INIT_LSTM;
    typedef LSTMLSTMHelper<STEER_INIT_LSTM, STEER_LSTM, 51> STEER_NN;

    typedef LSTMHelper<LSTMParams<3, 5>, FNNParams<8,10,1>, false> DELAY_LSTM;
    typedef LSTMHelper<LSTMParams<2, 60>, FNNParams<62, 100, 10>> DELAY_INIT_LSTM;
    typedef LSTMLSTMHelper<DELAY_INIT_LSTM, DELAY_LSTM, 51> DELAY_NN;

    typedef LSTMHelper<LSTMParams<10, 10>, FNNParams<20,20,3>, false> TERRA_LSTM;
    typedef LSTMHelper<LSTMParams<8, 60>, FNNParams<68, 100, 20>> TERRA_INIT_LSTM;
    typedef LSTMLSTMHelper<TERRA_INIT_LSTM, TERRA_LSTM, 51> TERRA_NN;

    struct SHARED_MEM_GRD_PARAMS {
        LSTMParams<5, 5> steer_lstm_params;
        FNNParams<10, 20, 1> steer_output_params;

        LSTMParams<3, 5> delay_lstm_params;
        FNNParams<8, 10, 1> delay_output_params;

        LSTMParams<10, 10> terra_lstm_params;
        FNNParams<20, 20, 3> terra_output_params;
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

    explicit RacerDoubleIntegratorKinematic(cudaStream_t stream = nullptr);
    explicit RacerDoubleIntegratorKinematic(std::string model_path, cudaStream_t stream = nullptr);

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
    TwoDTextureHelper<float>* getTextureHelper()
    {
        return tex_helper_;
    }

    STEER_LSTM* steer_network_d_ = nullptr;
    DELAY_LSTM* delay_network_d_ = nullptr;
    TERRA_LSTM* terra_network_d_ = nullptr;

protected:
    TwoDTextureHelper<float>* tex_helper_ = nullptr;

    std::shared_ptr<STEER_NN> steer_lstm_lstm_helper_;
    std::shared_ptr<DELAY_NN> delay_lstm_lstm_helper_;
    std::shared_ptr<TERRA_NN> terra_lstm_lstm_helper_;

};

#if __CUDACC__
#include "racer_double_integrator_kinematic.cu"
#endif

#endif  // MPPIGENERIC_DOUBLE_INTEGRATOR_KINEMATIC_CUH
