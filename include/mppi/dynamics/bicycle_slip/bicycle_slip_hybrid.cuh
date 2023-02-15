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

// TODO add input and output scaling into the lstm class itself

struct BicycleSlipHybridParams : public BicycleSlipKinematicParams
{
  float omega_v[2] = {1.0f, 1.1f};
  float ay_t[2] = {2.0f, 2.2f};
  float ay_b[2] = {4.0f, 4.4f};
  float ay_v[2] = {5.0f, 5.5f};
  float ay_angle = 6.0f;
};

class BicycleSlipHybrid : public BicycleSlipKinematicImpl<BicycleSlipHybrid, BicycleSlipHybridParams>
{
public:
    using PARENT_CLASS = BicycleSlipKinematicImpl<BicycleSlipHybrid, BicycleSlipHybridParams>;

    static const int SHARED_MEM_REQUEST_GRD = PARENT_CLASS::SHARED_MEM_REQUEST_GRD;
    static const int SHARED_MEM_REQUEST_BLK = PARENT_CLASS::SHARED_MEM_REQUEST_BLK;

    typedef typename PARENT_CLASS::state_array state_array;
    typedef typename PARENT_CLASS::control_array control_array;
    typedef typename PARENT_CLASS::output_array output_array;
    typedef typename PARENT_CLASS::dfdx dfdx;
    typedef typename PARENT_CLASS::dfdu dfdu;

    explicit BicycleSlipHybrid(cudaStream_t stream = nullptr);
    explicit BicycleSlipHybrid(std::string model_path, cudaStream_t stream = nullptr);

    std::string getDynamicsModelName() const override
    {
      return "Bicycle Slip Hybrid Model";
    }


    void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                           Eigen::Ref<state_array> state_der);

    __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);
protected:
};

#if __CUDACC__
#include "bicycle_slip_hybrid.cu"
#endif

#endif  // MPPIGENERIC_BICYCLE_SLIP_CUH
