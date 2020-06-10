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

  static const int BLOCKSIZE_WRX = 64;
  //NUM_ROLLOUTS has to be divisible by BLOCKSIZE_WRX
//  static const int NUM_ROLLOUTS = (NUM_ROLLOUTS/BLOCKSIZE_WRX)*BLOCKSIZE_WRX;
  static const int BLOCKSIZE_X = BDIM_X;
  static const int BLOCKSIZE_Y = BDIM_Y;
  static const int STATE_DIM = DYN_T::STATE_DIM;
  static const int CONTROL_DIM = DYN_T::CONTROL_DIM;

  // Number of samples per condition must be a multiple of the blockDIM
  static const int SAMPLES_PER_CONDITION = BDIM_X*SAMPLES_PER_CONDITION_MULTIPLIER;

  float value_func_threshold_ = 1000.0;

  int numTimesteps_;
  int optimizationStride_;

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
  RobustMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter, float gamma,
                       const Eigen::Ref<const StateCostWeight>& Q,
                       const Eigen::Ref<const Hessian>& Qf,
                       const Eigen::Ref<const ControlCostWeight>& R,
                       const Eigen::Ref<const control_array>& control_std_dev,
                       int num_timesteps = MAX_TIMESTEPS,
                       const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
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

  /*
  * @brief Resets the control commands to there initial values.
  */
//  void resetControls();

//  void cutThrottle();

//  void savitskyGolay();

//  void computeNominalTraj(const Eigen::Ref<const state_array>& state);

  /*void slideControlSeq(int stride);*/

// Update the importance sampler prior to calling computeControl
  void updateImportanceSampler(const Eigen::Ref<const state_array> &state, int stride);

  /**
  * @brief Compute the control given the current state of the system.
  * @param state The current state of the autorally system.
  */
  void computeControl(const Eigen::Ref<const state_array>& state) override {};

  control_trajectory getControlSeq() override {return nominal_control_trajectory_;};

  state_trajectory getStateSeq() override {return nominal_state_trajectory_;};

  // Does nothing. This reason is because the control sliding happens during the importance sampler update.
  // The control applied to the real system (during the MPPI rollouts) is the nominal control (which slides
  // during the importance sampler update), plus the feedback term. Inside the runControlIteration function
  // slideControl sequence is called prior to optimization, after the importance sampler update.
  void slideControlSequence(int steps) override {};

  // Feedback gain computation is done after the importance sampling update. The nominal trajectory computed
  // during the importance sampling update does not change after the optimization, thus the feedback gains will
  // not change either. In the current implementation of runControlIteration, the compute feedback gains is called
  // after the computation of the optimal control.
  void computeFeedbackGains(const Eigen::Ref<const state_array> state) override {};

  // TubeDiagnostics getTubeDiagnostics();

protected:
  bool importance_sampling_cuda_mem_init_ = false;
  int num_candidate_nominal_states_ = 9; // TODO should the initialization be a parameter?
  int best_index_ = 0;
  float nominal_normalizer_; ///< Variable for the normalizing term from sampling.
  int optimization_stride_; // Number of timesteps to apply the optimal control (== 1 for true MPC)
  int nominal_stride_ = 0; // Stride for the chosen nominal state of the importance sampler
  int real_stride_ = 0; // Stride for the optimal controller sliding
  bool nominal_state_init_ = false;


  OptimizerResult<ModelWrapperDDP<DYN_T>> last_result_;

//  TubeDiagnostics status_;

  // Storage classes
  control_trajectory nominal_control_trajectory_ = control_trajectory::Zero();
  state_trajectory nominal_state_trajectory_ = state_trajectory::Zero();

  // Make the control history size flexible, related to issue #30
  Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2> nominal_control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero(); // History used for nominal_state IS

  NominalCandidateVector candidate_nominal_states_ = {state_array::Zero()};
  Eigen::MatrixXf line_search_weights_; // At minimum there must be 3 candidates
  Eigen::MatrixXi importance_sampler_strides_; // Time index where control trajectory starts for each nominal state candidate
  Eigen::MatrixXf candidate_trajectory_costs_;
  Eigen::MatrixXf candidate_free_energy_;
  std::vector<float> feedback_gain_vector_;

  void allocateCUDAMemory();

  void deallocateCUDAMemory();

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
  void computeNominalFeedbackGains();

  // CUDA Memory
  float* importance_sampling_states_d_;
  float* importance_sampling_costs_d_;
  int* importance_sampling_strides_d_;
  float* feedback_gain_array_d_;

  float* nominal_state_d_;

  // control_noise_d_ is also used to hold the rollout noise for the quick estimate free energy.
  // Here num_candidates*num_samples_per_condition < 2*num_rollouts. -> we should enforce this

  //  // Previous storage classes
//  std::vector<float> U_;
//  std::vector<float> U_optimal_;
//  std::vector<float> augmented_nominal_costs_; ///< Array of the trajectory costs.
//  std::vector<float> augmented_real_costs_; ///< Array of the trajectory costs.
//  std::vector<float> pure_real_costs_; ///< Array of the trajectory costs.
//
//  std::vector<float> state_solution_; ///< Host array for keeping track of the nomimal trajectory.
//  std::vector<float> control_solution_;
//  std::vector<float> importance_hist_;
//  std::vector<float> optimal_control_hist_;
//  std::vector<float> du_; ///< Host array for computing the optimal control update.
//  std::vector<float> nu_;
//  std::vector<float> init_u_;
//  std::vector<float> feedback_gains_;
//
//  float* feedback_gains_d_;
//  float *augmented_nominal_costs_d_, *augmented_real_costs_d_, *pure_real_costs_d_;
//  float *state_d_, *nominal_state_d_;
//  float *U_d_;
//  float *nu_d_;
//  float *du_d_;
//  float *dx_d_;

};

//template <class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y, int SAMPLES_PER_CONDITION>
//__global__ void initEvalKernel(DYN_T* dynamics, COST_T* costs, float dt,
//    int num_timesteps, float* init_states_d, float* strides_d,
//    float* u_d, float* du_d, float* sigma_u_d, float* trajectory_costs_d);
//
//template<class DYN_T, class COST_T>
//void launchInitEvalKernel(DYN_T* dynamics, COST_T* costs, float dt,
//    int num_timesteps, float* x_d, float* u_d, float* du_d,
//    float* sigma_u_d, float* candidate_trajectory_costs, cudaStream_t stream);

#if __CUDACC__
#include "robust_mppi_controller.cu"
#endif

#endif /* MPPI_GEN_RMPPI_CONTROLLER_CUH_ */