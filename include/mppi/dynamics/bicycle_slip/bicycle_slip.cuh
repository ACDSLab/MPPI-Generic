//
// Created by jason on 12/12/22.
//

#ifndef MPPIGENERIC_BICYCLE_SLIP_CUH
#define MPPIGENERIC_BICYCLE_SLIP_CUH

#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/utils/angle_utils.cuh>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>
#include "mppi/utils/texture_helpers/two_d_texture_helper.cuh"
#include "bicycle_slip_kinematic.cuh"

struct BicycleSlipParams : public BicycleSlipKinematicParams
{
  float wheel_angle_scale = -9.2f;
};

// TODO you have incorrect template on the init vs the predictor

class BicycleSlip : public MPPI_internal::Dynamics<BicycleSlip, BicycleSlipParams>
{
public:
    using PARENT_CLASS = MPPI_internal::Dynamics<BicycleSlip, BicycleSlipParams>;
    typedef LSTMHelper<LSTMParams<5, 5>, FNNParams<10,5,1>, false> STEER_LSTM;
    typedef LSTMHelper<LSTMParams<4, 200>, FNNParams<204, 2000, 10>> STEER_INIT_LSTM;
    typedef LSTMLSTMHelper<STEER_INIT_LSTM, STEER_LSTM, 51> STEER_NN;

    typedef LSTMHelper<LSTMParams<3, 5>, FNNParams<8,10,1>, false> DELAY_LSTM;
    typedef LSTMHelper<LSTMParams<2, 200>, FNNParams<202, 2000, 10>> DELAY_INIT_LSTM;
    typedef LSTMLSTMHelper<DELAY_INIT_LSTM, DELAY_LSTM, 51> DELAY_NN;

    typedef LSTMHelper<LSTMParams<10, 10>, FNNParams<20,20,4>, false> TERRA_LSTM;
    typedef LSTMHelper<LSTMParams<10, 200>, FNNParams<210, 2000, 20>> TERRA_INIT_LSTM;
    typedef LSTMLSTMHelper<TERRA_INIT_LSTM, TERRA_LSTM, 51> TERRA_NN;

    struct SHARED_MEM_GRD_PARAMS {
        LSTMParams<5, 5> steer_lstm_params;
        FNNParams<10, 5, 1> steer_output_params;

        LSTMParams<3, 5> delay_lstm_params;
        FNNParams<8, 10, 1> delay_output_params;

        LSTMParams<10, 10> terra_lstm_params;
        FNNParams<20, 20, 4> terra_output_params;
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

    explicit BicycleSlip(cudaStream_t stream = nullptr);
    explicit BicycleSlip(std::string model_path, cudaStream_t stream = nullptr);

    std::string getDynamicsModelName() const override
    {
      return "Bicycle Slip Model";
    }

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

    __device__ void updateState(float* state, float* next_state, float* state_der, const float dt) {}
    __device__ void updateState(float* state, float* next_state, float* state_der, const float dt, DYN_PARAMS_T* params_p);
    __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);
    __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                                float* theta_s, const float t, const float dt);

    // state_array interpolateState(const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2,
    //                              const float alpha);

    // bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    //                  Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);


    void getStoppingControl(const Eigen::Ref<const state_array>& state, Eigen::Ref<control_array> u);

    Eigen::Quaternionf attitudeFromState(const Eigen::Ref<const state_array>& state);
    Eigen::Vector3f positionFromState(const Eigen::Ref<const state_array>& state);
    Eigen::Vector3f velocityFromState(const Eigen::Ref<const state_array>& state);
    Eigen::Vector3f angularRateFromState(const Eigen::Ref<const state_array>& state);
    state_array stateFromOdometry(const Eigen::Quaternionf& q, const Eigen::Vector3f& pos, const Eigen::Vector3f& vel,
                                  const Eigen::Vector3f& omega);

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
#include "bicycle_slip.cu"
#endif

#endif  // MPPIGENERIC_BICYCLE_SLIP_CUH
