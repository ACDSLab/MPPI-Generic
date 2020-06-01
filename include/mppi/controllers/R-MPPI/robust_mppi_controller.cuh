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

  // using m_dyn = typename ModelWrapperDDP<DYN_T>::Scalar;
  using FeedbackGainTrajectory = typename util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM>;
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

  bool nominalStateInit_ = false;
  int numTimesteps_;
  int hz_;
  int optimizationStride_;


  //Define DDP optimizer for computing feedback gains around MPPI solution
  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;
  StateCostWeight Q_;
  Hessian Qf_;
  ControlCostWeight R_;
  control_array U_MIN_; // Moved to Dynamics
  control_array U_MAX_; // Moved to Dynamics

  state_array nominal_state_;

  /**
  * @brief Constructor for mppi controller class.
  * @param num_timesteps The number of timesteps to look ahead for.
  * @param dt The time increment. horizon = num_timesteps*dt
  * @param model A basis function model of the system dynamics.
  * @param cost An MppiCost object.
  * @param mppi_node Handle to a ros node with mppi parameters available as ros params.
  */
//  RobustMPPIController(DYN_T* model, COST_T* cost, int num_timesteps,
//                       int hz, float gamma,
//                       float* exploration_var, float* init_control,
//                       int num_optimization_iters = 1,
//                       int opt_stride = 1, cudaStream_t = 0);
  RobustMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter, float gamma,
                       const Eigen::Ref<const control_array>& control_variance,
                       int num_timesteps,
                       const Eigen::Ref<const control_trajectory>& init_control_traj,
                       cudaStream_t stream);

  /**
  * @brief Destructor for mppi controller class.
  */
  ~RobustMPPIController();

//  void computeFeedbackGains(const Eigen::Ref<const state_array>& s) override;

  FeedbackGainTrajectory getFeedbackGains() { return result_.feedback_gain;};

  /*
  * @brief Resets the control commands to there initial values.
  */
//  void resetControls();

//  void cutThrottle();

//  void savitskyGolay();

//  void computeNominalTraj(const Eigen::Ref<const state_array>& state);

  /*void slideControlSeq(int stride);*/

//  void updateImportanceSampler(const Eigen::Ref<const state_array>& state, int stride);

  /**
  * @brief Compute the control given the current state of the system.
  * @param state The current state of the autorally system.
  */
  void computeControl(const Eigen::Ref<const state_array>& state) override {};

  control_trajectory getControlSeq() override {return nominal_control_trajectory;};

  state_trajectory getStateSeq() override {return nominal_state_trajectory;};

  void slideControlSequence(int steps) override {};

  // TubeDiagnostics getTubeDiagnostics();

protected:
  bool importance_sampling_cuda_mem_init = false;
  int num_candidate_nominal_states = 9; // TODO should the initialization be a parameter?
  int best_index = 0;
  float normalizer_, nominal_normalizer_; ///< Variable for the normalizing term from sampling.

  OptimizerResult<ModelWrapperDDP<DYN_T>> result_;
  OptimizerResult<ModelWrapperDDP<DYN_T>> last_result_;

//  TubeDiagnostics status_;

  // Storage classes
  control_trajectory nominal_control_trajectory = control_trajectory::Zero();
  state_trajectory nominal_state_trajectory = state_trajectory::Zero();

  NominalCandidateVector candidate_nominal_states = {state_array::Zero()};
  Eigen::MatrixXf line_search_weights; // At minimum there must be 3 candidates
  Eigen::MatrixXi importance_sampler_strides; // Time index where control trajectory starts for each nominal state candidate
  Eigen::MatrixXf candidate_trajectory_costs;
  Eigen::MatrixXf candidate_free_energy;


  void allocateCUDAMemory();

  void deallocateNominalStateCandidateMemory();

  // Initializes the num_candidates, candidate_nominal_states, linesearch_weights,
  // and allocates the associated CUDA memory
  void updateNumCandidates(int new_num_candidates);

  void resetCandidateCudaMem();

  void getInitNominalStateCandidates(
        const Eigen::Ref<const state_array>& nominal_x_k,
        const Eigen::Ref<const state_array>& nominal_x_kp1,
        const Eigen::Ref<const state_array>& real_x_kp1);

  // compute the line search weights
  void computeLineSearchWeights();

  // compute the importance sampler strides
  void computeImportanceSamplerStride(int stride);

  // Compute the baseline of the candidates
  float computeCandidateBaseline();

  // Get the best index based on the candidate free energy
  void computeBestIndex();

  // CUDA Memory
  float* importance_sampling_states_d_;
  float* importance_sampling_costs_d_;
  float* importance_sampling_strides_d_;
  float* trajectory_costs_d_;


  float* initial_state_d_;
  float* control_d_;
  float* state_d_;
  float* nominal_state_d_;
  float* control_variance_d_;
  float* control_noise_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS * 2
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