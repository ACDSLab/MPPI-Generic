/*
* Software License Agreement (BSD License)
* Copyright (c) 2013, Georgia Institute of Technology
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/**********************************************
 * @file mppi_controller.cuh
 * @author Grady Williams <gradyrw@gmail.com>
 * @date May 24, 2017
 * @copyright 2017 Georgia Institute of Technology
 * @brief Class definition for the MPPI controller.
 ***********************************************/

#ifndef MPPI_GEN_RMPPI_CONTROLLER_CUH_
#define MPPI_GEN_RMPPI_CONTROLLER_CUH_

#include <curand.h>
#include <mppi/controllers/controller.cuh>
#include <mppi/core/mppi_common.cuh>

template <class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS = 2560,
          int BDIM_X = 64, int BDIM_Y = 1, int SAMPLES_PER_CONDITION_MULTIPLIER = 1>
class RobustMPPIController : public Controller<DYN_T, COST_T,
                                            MAX_TIMESTEPS,
                                            NUM_ROLLOUTS,
                                            BDIM_X,
                                            BDIM_Y> {

public:

  /**
   * Set up useful types
   */
  using control_array = typename Controller<DYN_T, COST_T,
                                            MAX_TIMESTEPS,
                                            NUM_ROLLOUTS,
                                            BDIM_X,
                                            BDIM_Y>::control_array;

  using control_trajectory = typename Controller<DYN_T, COST_T,
                                                 MAX_TIMESTEPS,
                                                 NUM_ROLLOUTS,
                                                 BDIM_X,
                                                 BDIM_Y>::control_trajectory;

  using state_trajectory = typename Controller<DYN_T, COST_T,
                                               MAX_TIMESTEPS,
                                               NUM_ROLLOUTS,
                                               BDIM_X,
                                               BDIM_Y>::state_trajectory;

  using state_array = typename Controller<DYN_T, COST_T,
                                          MAX_TIMESTEPS,
                                          NUM_ROLLOUTS,
                                          BDIM_X,
                                          BDIM_Y>::state_array;

  using sampled_cost_traj = typename Controller<DYN_T, COST_T,
                                                MAX_TIMESTEPS,
                                                NUM_ROLLOUTS,
                                                BDIM_X,
                                                BDIM_Y>::sampled_cost_traj;

  using feedback_gain_trajectory = typename Controller<DYN_T, COST_T,
                                                MAX_TIMESTEPS,
                                                NUM_ROLLOUTS,
                                                BDIM_X,
                                                BDIM_Y>::feedback_gain_trajectory;

  // using m_dyn = typename ModelWrapperDDP<DYN_T>::Scalar;
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;
  using NominalCandidateVector = typename util::NamedEigenAlignedVector<state_array>;

  static const int BLOCKSIZE_X = BDIM_X;
  static const int BLOCKSIZE_Y = BDIM_Y;
  static const int STATE_DIM = DYN_T::STATE_DIM;
  static const int CONTROL_DIM = DYN_T::CONTROL_DIM;

  // Number of samples per condition must be a multiple of the blockDIM
  static const int SAMPLES_PER_CONDITION = BDIM_X*SAMPLES_PER_CONDITION_MULTIPLIER;

  float value_function_threshold_ = 1000.0;

  state_array nominal_state_ = state_array::Zero();

  /**
  * @brief Constructor for mppi controller class
  * @param model A basis function model of the system dynamics.
  * @param cost An MppiCost object.
  * @param dt The time increment. horizon = num_timesteps*dt
  * @param max_iter number of times to repeat control sequence calculations
  * @param gamma
  * @param num_timesteps The number of timesteps to look ahead for.
  * TODO Finish this description
  */
  RobustMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                       float lambda, float alpha,
                       float value_function_threshold,
                       const Eigen::Ref<const StateCostWeight>& Q,
                       const Eigen::Ref<const Hessian>& Qf,
                       const Eigen::Ref<const ControlCostWeight>& R,
                       const Eigen::Ref<const control_array>& control_std_dev,
                       int num_timesteps = MAX_TIMESTEPS,
                       const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                       int num_candidate_nominal_states = 9,
                       int optimization_stride = 1,
                       cudaStream_t stream = nullptr);

  /**
  * @brief Destructor for mppi controller class.
  */
  ~RobustMPPIController();

  feedback_gain_trajectory getFeedbackGains() override {
    return this->result_.feedback_gain;
  };

  // Initializes the num_candidates, candidate_nominal_states, linesearch_weights,
  // and allocates the associated CUDA memory
  void updateNumCandidates(int new_num_candidates);


  // Update the importance sampler prior to calling computeControl
  void updateImportanceSamplingControl(const Eigen::Ref<const state_array> &state, int stride);

  /**
  * @brief Compute the control given the current state of the system.
  * @param state The current state of the autorally system.
  */
  void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride=1);

  control_trajectory getControlSeq() override {return this->control_;};

  state_trajectory getStateSeq() override {return nominal_state_trajectory_;};

  state_trajectory getAncillaryStateSeq() {return this->result_.state_trajectory;};

  // Does nothing. This reason is because the control sliding happens during the importance sampler update.
  // The control applied to the real system (during the MPPI rollouts) is the nominal control (which slides
  // during the importance sampler update), plus the feedback term. Inside the runControlIteration function
  // slideControl sequence is called prior to optimization, after the importance sampler update.
  void slideControlSequence(int steps) override {};

  // Feedback gain computation is done after the importance sampling update. The nominal trajectory computed
  // during the importance sampling update does not change after the optimization, thus the feedback gains will
  // not change either. In the current implementation of runControlIteration, the compute feedback gains is called
  // after the computation of the optimal control.
  void computeFeedbackGains(const Eigen::Ref<const state_array>& state) override {};

  Eigen::MatrixXf getCandidateFreeEnergy() {return candidate_free_energy_;};

protected:
  bool importance_sampling_cuda_mem_init_ = false;
  int num_candidate_nominal_states_;
  int best_index_ = 0;  // Selected nominal state candidate
  int optimization_stride_; // Number of timesteps to apply the optimal control (== 1 for true MPC)
  int nominal_stride_ = 0; // Stride for the chosen nominal state of the importance sampler
  int real_stride_ = 0; // Stride for the optimal controller sliding
  bool nominal_state_init_ = false;
  float baseline_nominal_ = 100.0; // Cost baseline for the nominal state
  float normalizer_nominal_ = 100.0;  // Normalizer variable for the nominal state

  // Free energy variables
  float nominal_free_energy_mean_ = 0;
  float nominal_free_energy_variance_ = 0;
  float nominal_free_energy_modified_variance_ = 0;

  // Storage classes
  control_trajectory nominal_control_trajectory_ = control_trajectory::Zero();
  state_trajectory nominal_state_trajectory_ = state_trajectory::Zero();
  sampled_cost_traj trajectory_costs_nominal_ = sampled_cost_traj::Zero();

  // Make the control history size flexible, related to issue #30
  Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2> nominal_control_history_; // History used for nominal_state IS

  NominalCandidateVector candidate_nominal_states_ = {state_array::Zero()};
  Eigen::MatrixXf line_search_weights_; // At minimum there must be 3 candidates
  Eigen::MatrixXi importance_sampler_strides_; // Time index where control trajectory starts for each nominal state candidate
  Eigen::MatrixXf candidate_trajectory_costs_;
  Eigen::MatrixXf candidate_free_energy_;
  std::vector<float> feedback_gain_vector_;

  void allocateCUDAMemory();

  void deallocateCUDAMemory();

  void copyNominalControlToDevice();

  void deallocateNominalStateCandidateMemory();

  void resetCandidateCudaMem();

  void getInitNominalStateCandidates(
        const Eigen::Ref<const state_array>& nominal_x_k,
        const Eigen::Ref<const state_array>& nominal_x_kp1,
        const Eigen::Ref<const state_array>& real_x_kp1);

  // compute the line search weights
  void computeLineSearchWeights();

  void computeNominalStateAndStride(const Eigen::Ref<const state_array> &state, int stride);

  // compute the importance sampler strides
  void computeImportanceSamplerStride(int stride);

  // Compute the baseline of the candidates
  float computeCandidateBaseline();

  // Get the best index based on the candidate free energy
  void computeBestIndex();

  // Computes and saves the feedback gains used in the rollout kernel and tracking.
  void computeNominalFeedbackGains(const Eigen::Ref<const state_array> &state);

  // CUDA Memory
  float* importance_sampling_states_d_;
  float* importance_sampling_costs_d_;
  int* importance_sampling_strides_d_;
  float* feedback_gain_array_d_;
};

#if __CUDACC__
#include "robust_mppi_controller.cu"
#endif

#endif /* MPPI_GEN_RMPPI_CONTROLLER_CUH_ */