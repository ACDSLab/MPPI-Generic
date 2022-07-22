//
// Created by jason on 10/30/19.
//

#ifndef MPPIGENERIC_CONTROLLER_CUH
#define MPPIGENERIC_CONTROLLER_CUH

#include <array>
#include <Eigen/Core>
#include <chrono>
#include "curand.h"

#include <mppi/core/mppi_common.cuh>
#include <mppi/feedback_controllers/feedback.cuh>
#include <mppi/utils/gpu_err_chk.cuh>
#include <mppi/utils/math_utils.h>

#include <cfloat>
#include <utility>

struct freeEnergyEstimate
{
  float increase = -1;
  float previousBaseline = -1;
  float freeEnergyMean = -1;
  float freeEnergyVariance = -1;
  float freeEnergyModifiedVariance = -1;
  float normalizerPercent = -1;
};

struct MPPIFreeEnergyStatistics
{
  int nominal_state_used = 0;

  freeEnergyEstimate nominal_sys;
  freeEnergyEstimate real_sys;
};

template <int S_DIM, int C_DIM, int MAX_TIMESTEPS>
struct ControllerParams
{
  static const int TEMPLATED_STATE_DIM = S_DIM;
  static const int TEMPLATED_CONTROL_DIM = C_DIM;
  static const int TEMPLATED_MAX_TIMESTEPS = MAX_TIMESTEPS;
  float dt_;
  float lambda_;       // Value of the temperature in the softmax.
  float alpha_ = 0.0;  //
  // MAX_TIMESTEPS is defined as an upper bound, if lower that region is just ignored when calculating control
  // does not reallocate cuda memory
  int num_timesteps_ = MAX_TIMESTEPS;
  int num_iters_ = 1;  // Number of optimization iterations
  unsigned seed_ = std::chrono::system_clock::now().time_since_epoch().count();

  Eigen::Matrix<float, C_DIM, 1> control_std_dev_ = Eigen::Matrix<float, C_DIM, 1>::Zero();
  Eigen::Matrix<float, C_DIM, MAX_TIMESTEPS> init_control_traj_ = Eigen::Matrix<float, C_DIM, MAX_TIMESTEPS>::Zero();
  Eigen::Matrix<float, C_DIM, 1> slide_control_scale_ = Eigen::Matrix<float, C_DIM, 1>::Zero();
};

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T = ControllerParams<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, MAX_TIMESTEPS>>
class Controller
{
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * typedefs for access to templated class from outside classes
   */
  typedef DYN_T TEMPLATED_DYNAMICS;
  typedef COST_T TEMPLATED_COSTS;
  typedef FB_T TEMPLATED_FEEDBACK;
  typedef PARAMS_T TEMPLATED_PARAMS;
  using TEMPLATED_FEEDBACK_STATE = typename FB_T::TEMPLATED_FEEDBACK_STATE;
  using TEMPLATED_FEEDBACK_PARAMS = typename FB_T::TEMPLATED_PARAMS;
  using TEMPLATED_FEEDBACK_GPU = typename FB_T::TEMPLATED_GPU_FEEDBACK;
  static const int TEMPLATED_FEEDBACK_TIMESTEPS = FB_T::FB_TIMESTEPS;

  /**
   * Aliases
   */
  // Control typedefs
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, MAX_TIMESTEPS> control_trajectory;  // A control trajectory

  // State typedefs
  using state_array = typename DYN_T::state_array;
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM, MAX_TIMESTEPS> state_trajectory;  // A state trajectory

  // Output typedefs
  using output_array = typename DYN_T::output_array;
  typedef Eigen::Matrix<float, DYN_T::OUTPUT_DIM, MAX_TIMESTEPS> output_trajectory;  // An output trajectory

  // Cost typedefs
  typedef Eigen::Matrix<float, MAX_TIMESTEPS + 1, 1> cost_trajectory;  // +1 for terminal cost
  typedef Eigen::Matrix<float, NUM_ROLLOUTS, 1> sampled_cost_traj;
  typedef Eigen::Matrix<int, MAX_TIMESTEPS, 1> crash_status_trajectory;

  Controller(DYN_T* model, COST_T* cost, FB_T* fb_controller, float dt, int max_iter, float lambda, float alpha,
             const Eigen::Ref<const control_array>& control_std_dev, int num_timesteps = MAX_TIMESTEPS,
             const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
             cudaStream_t stream = nullptr)
  {
    // Create the random number generator
    createAndSeedCUDARandomNumberGen();
    model_ = model;
    cost_ = cost;
    fb_controller_ = fb_controller;
    params_.dt_ = dt;
    params_.num_iters_ = max_iter;
    params_.lambda_ = lambda;
    params_.alpha_ = alpha;
    params_.num_timesteps_ = num_timesteps;

    params_.control_std_dev_ = control_std_dev;
    params_.init_control_traj_ = init_control_traj;
    control_ = init_control_traj;
    control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

    // Bind the model and control to the given stream
    setCUDAStream(stream);
    // Create new stream for visualization purposes
    HANDLE_ERROR(cudaStreamCreate(&vis_stream_));

    // Call the GPU setup functions of the model, cost and feedback controller
    model_->GPUSetup();
    cost_->GPUSetup();
    fb_controller_->GPUSetup();

    /**
     * When implementing your own version make sure to write your own allocateCUDAMemory and call it from the
     * constructor along with any other methods to copy memory to the device and back
     */
    // TODO pass function pointer?
  }

  Controller(DYN_T* model, COST_T* cost, FB_T* fb_controller, PARAMS_T& params, cudaStream_t stream = nullptr)
  {
    model_ = model;
    cost_ = cost;
    fb_controller_ = fb_controller;
    // Create the random number generator
    createAndSeedCUDARandomNumberGen();
    setParams(params);
    control_ = params_.init_control_traj_;
    control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

    // Bind the model and control to the given stream
    setCUDAStream(stream);
    // Create new stream for visualization purposes
    HANDLE_ERROR(cudaStreamCreate(&vis_stream_));

    // Call the GPU setup functions of the model, cost and feedback controller
    model_->GPUSetup();
    cost_->GPUSetup();
    fb_controller_->GPUSetup();

    /**
     * When implementing your own version make sure to write your own allocateCUDAMemory and call it from the
     * constructor along with any other methods to copy memory to the device and back
     */
    // TODO pass function pointer?
  }

  // TODO should be private with test as a friend to ensure it is only used in testing
  Controller() = default;

  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a basePlant reference
   */
  virtual ~Controller()
  {
    // Free the CUDA memory of every object
    model_->freeCudaMem();
    cost_->freeCudaMem();
    fb_controller_->freeCudaMem();

    // Free the CUDA memory of the controller
    deallocateCUDAMemory();
  }

  // =================== METHODS THAT SHOULD HAVE NO DEFAULT ==========================
  // ======== PURE VIRTUAL =========
  /**
   * Given a state, calculates the optimal control sequence using MPPI according
   * to the cost function used as part of the template
   * @param state - the current state from which we would like to calculate
   * a control sequence
   */
  virtual void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride) = 0;

  /**
   * @param steps dt's to slide control sequence forward
   * Slide the control sequence forwards steps steps
   */
  virtual void slideControlSequence(int optimization_stride) = 0;

  /**
   * Call a kernel to evaluate the sampled state trajectories for visualization
   * and debugging.
   */
  virtual void calculateSampledStateTrajectories() = 0;

  // ================ END OF METHODS WITH NO DEFAULT =============
  // ======== PURE VIRTUAL END =====

  virtual std::string getControllerName()
  {
    return "name not set";
  };
  virtual std::string getCostFunctionName()
  {
    return cost_->getCostFunctionName();
  }

  virtual void initFeedback()
  {
    enable_feedback_ = true;
    fb_controller_->initTrackingController();
  };

  virtual std::vector<state_trajectory> getSampledStateTrajectories() const
  {
    return sampled_trajectories_;
  }

  virtual std::vector<cost_trajectory> getSampledCostTrajectories() const
  {
    return sampled_costs_;
  }

  virtual std::vector<crash_status_trajectory> getSampledCrashStatusTrajectories() const
  {
    return sampled_crash_status_;
  }

  virtual std::vector<float> getTopTransformedCosts() const
  {
    return top_n_costs_;
  }

  /**
   * only used in rmppi, here for generic calls in base_plant. Jank as hell
   * @param state
   * @param stride
   */
  void updateImportanceSamplingControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
  {
  }

  /**
   * Used to update the importance sampler
   * @param nominal_control the new nominal control sequence to sample around
   */
  virtual void updateImportanceSampler(const Eigen::Ref<const control_trajectory>& nominal_control)
  {
    // TODO copy to device new control sequence
    control_ = nominal_control;
  }

  /**
   * determines the control that should
   * @param state
   * @param rel_time
   * @return
   */
  virtual control_array getCurrentControl(state_array& state, double rel_time, state_array& target_nominal_state,
                                          control_trajectory& c_traj, TEMPLATED_FEEDBACK_STATE& fb_state)
  {
    // MPPI control
    control_array u_ff = interpolateControls(rel_time, c_traj);
    control_array u_fb = control_array::Zero();
    if (enable_feedback_)
    {
      u_fb = interpolateFeedback(state, target_nominal_state, rel_time, fb_state);
    }
    control_array result = u_ff + u_fb;
    // printf("rel_time %f\n", rel_time);
    // printf("uff: %f, %f u_fb: %f, %f\n", u_ff[0], u_ff[1], u_fb[0], u_fb[1]);

    // TODO this is kinda jank
    state_array empty_state = state_array::Zero();
    model_->enforceConstraints(empty_state, result);

    return result;
  }

  /**
   * determines the interpolated control from control_seq_, linear interpolation
   * @param rel_time time since the solution was calculated
   * @return
   */
  virtual control_array interpolateControls(double rel_time, control_trajectory& c_traj)
  {
    int lower_idx = (int)(rel_time / getDt());
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * getDt()) / getDt();

    control_array interpolated_control;
    control_array prev_cmd = c_traj.col(lower_idx);
    control_array next_cmd = c_traj.col(upper_idx);
    interpolated_control = (1 - alpha) * prev_cmd + alpha * next_cmd;

    // printf("prev: %d %f, %f\n", lower_idx, prev_cmd[0], prev_cmd[1]);
    // printf("next: %d %f, %f\n", upper_idx, next_cmd[0], next_cmd[1]);
    // printf("smoother: %f\n", alpha);
    return interpolated_control;
  }

  virtual state_array interpolateState(state_trajectory& s_traj, double rel_time)
  {
    int lower_idx = (int)(rel_time / getDt());
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * getDt()) / getDt();

    return model_->interpolateState(s_traj.col(lower_idx), s_traj.col(upper_idx), alpha);
  }

  /**
   *
   * @param state
   * @param rel_time
   * @return
   */
  virtual control_array interpolateFeedback(state_array& state, state_array& target_nominal_state, double rel_time,
                                            TEMPLATED_FEEDBACK_STATE& fb_state)
  {
    return fb_controller_->interpolateFeedback_(state, target_nominal_state, rel_time, fb_state);
  }

  /**
   * returns the current control sequence
   */
  virtual control_trajectory getControlSeq() const
  {
    return control_;
  };

  /**
   * Gets the state sequence of the nominal trajectory
   */
  virtual state_trajectory getTargetStateSeq() const
  {
    return state_;
  }

  /**
   * Return all the sampled costs sequences
   */
  virtual sampled_cost_traj getSampledCostSeq() const
  {
    return trajectory_costs_;
  };

  /**
   * Return control feedback gains
   */
  // TODO: Think of a better name for this method?
  virtual TEMPLATED_FEEDBACK_STATE getFeedbackState() const
  {
    if (enable_feedback_)
    {
      return fb_controller_->getFeedbackState();
    }
    else
    {
      TEMPLATED_FEEDBACK_STATE default_state;
      return default_state;
    }
  };

  virtual TEMPLATED_FEEDBACK_PARAMS getFeedbackParams() const
  {
    if (enable_feedback_)
    {
      return fb_controller_->getParams();
    }
    else
    {
      TEMPLATED_FEEDBACK_PARAMS default_fb_params;
      return default_fb_params;
    }
  }

  // Indicator for algorithm health, should be between 0.01 and 0.1 anecdotally
  float getNormalizerPercent() const
  {
    return this->getNormalizerCost() / (float)NUM_ROLLOUTS;
  }

  /**
   * Computes the actual trajectory given the MPPI optimal control and the
   * feedback gains computed by DDP. If feedback is not enabled, then we return
   * zero since this function would not make sense.
   */
  virtual void computeFeedbackPropagatedStateSeq()
  {
    if (!enable_feedback_)
    {
      return;
    }
    // Compute the nominal trajectory
    propagated_feedback_state_trajectory_.col(0) = getActualStateSeq().col(0);  // State that we optimized from
    state_array xdot;
    state_array current_state, next_state;
    output_array output;
    control_array current_control;
    for (int i = 0; i < getNumTimesteps() - 1; ++i)
    {
      current_state = propagated_feedback_state_trajectory_.col(i);
      // MPPI control apply feedback at the given timestep against the nominal trajectory at that timestep
      current_control = getControlSeq().col(i) + getFeedbackControl(current_state, getTargetStateSeq().col(i), i);
      model_->step(current_state, next_state, xdot, current_control, output, i, getDt());
      propagated_feedback_state_trajectory_.col(i + 1) = next_state;
    }
  }

  /**
   *
   * @return State trajectory from optimized state with MPPI control and computed feedback gains
   */
  state_trajectory getFeedbackPropagatedStateSeq() const
  {
    return propagated_feedback_state_trajectory_;
  };

  control_array getControlStdDev() const
  {
    return params_.control_std_dev_;
  };

  float getBaselineCost(int ind = 0) const
  {
    return cost_baseline_and_norm_[ind].x;
  };
  float getNormalizerCost(int ind = 0) const
  {
    return cost_baseline_and_norm_[ind].y;
  };

  /**
   * returns the current state sequence
   */
  state_trajectory getActualStateSeq() const
  {
    return state_;
  };

  virtual void computeFeedbackHelper(const Eigen::Ref<const state_array>& state,
                                     const Eigen::Ref<const state_trajectory>& state_traj,
                                     const Eigen::Ref<const control_trajectory>& control_traj)
  {
    if (!enable_feedback_)
    {
      return;
    }
    fb_controller_->computeFeedback(state, state_traj, control_traj);
  }

  virtual void computeFeedback(const Eigen::Ref<const state_array>& state)
  {
    computeFeedbackHelper(state, getTargetStateSeq(), getControlSeq());
  }

  virtual control_array getFeedbackControl(const Eigen::Ref<const state_array>& state,
                                           const Eigen::Ref<const state_array>& goal_state, int t)
  {
    return fb_controller_->k(state, goal_state, t);
  }

  void smoothControlTrajectoryHelper(Eigen::Ref<control_trajectory> u,
                                     const Eigen::Ref<Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>>& control_history)
  {
    // TODO generalize to any size filter
    // TODO does the logic of handling control history reasonable?

    // Create the filter coefficients
    Eigen::Matrix<float, 1, 5> filter_coefficients;
    filter_coefficients << -3, 12, 17, 12, -3;
    filter_coefficients /= 35.0;

    // Create and fill a control buffer that we can apply the convolution filter
    Eigen::MatrixXf control_buffer(getNumTimesteps() + 4, DYN_T::CONTROL_DIM);

    // Fill the first two timesteps with the control history
    control_buffer.topRows(2) = control_history.transpose();

    // Fill the center timesteps with the current nominal trajectory
    control_buffer.middleRows(2, getNumTimesteps()) = u.transpose();

    // Fill the last two timesteps with the end of the current nominal control trajectory
    control_buffer.row(getNumTimesteps() + 2) = u.transpose().row(getNumTimesteps() - 1);
    control_buffer.row(getNumTimesteps() + 3) = u.transpose().row(getNumTimesteps() - 1);

    // Apply convolutional filter to each timestep
    for (int i = 0; i < getNumTimesteps(); ++i)
    {
      u.col(i) = (filter_coefficients * control_buffer.middleRows(i, 5)).transpose();
    }
  }

  virtual void slideControlSequenceHelper(int steps, Eigen::Ref<control_trajectory> u)
  {
    for (int i = 0; i < getNumTimesteps(); ++i)
    {
      int ind = std::min(i + steps, getNumTimesteps() - 1);
      u.col(i) = u.col(ind);
      if (i + steps > getNumTimesteps() - 1)
      {
        u.col(i) = (u.col(ind).array() - model_->zero_control_.array()) * params_.slide_control_scale_.array() +
                   model_->zero_control_.array();
      }
    }
  }

  virtual void saveControlHistoryHelper(int steps, const Eigen::Ref<const control_trajectory>& u_trajectory,
                                        Eigen::Ref<Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>> u_history)
  {
    if (steps == 1)
    {  // We only moved one timestep
      u_history.col(0) = u_history.col(1);
      u_history.col(1) = u_trajectory.col(0);
    }
    else if (steps >= 2)
    {  // We have moved more than one timestep, but our history size is still only 2
      u_history.col(0) = u_trajectory.col(steps - 2);
      u_history.col(1) = u_trajectory.col(steps - 1);
    }
  }

  /**
   * Reset Controls
   */
  virtual void resetControls(){
    // TODO
  };

  virtual void computeStateTrajectoryHelper(Eigen::Ref<state_trajectory> result,
                                            const Eigen::Ref<const state_array>& x0,
                                            const Eigen::Ref<const control_trajectory>& u)
  {
    result.col(0) = x0;
    state_array xdot;
    state_array state, next_state;
    output_array output;
    model_->initializeDynamics(state.col(0), u.col(0), 0, getDt());
    for (int i = 0; i < getNumTimesteps() - 1; ++i)
    {
      state = result.col(i);
      control_array u_i = u.col(i);
      model_->enforceConstraints(state, u_i);
      model_->step(state, next_state, xdot, u_i, output, i, getDt());
      result.col(i + 1) = next_state;
    }
  }

  void setNumTimesteps(int num_timesteps)
  {
    // TODO fix the tracking controller as well
    if ((num_timesteps <= MAX_TIMESTEPS) && (num_timesteps > 0))
    {
      params_.num_timesteps_ = num_timesteps;
    }
    else
    {
      params_.num_timesteps_ = MAX_TIMESTEPS;
      printf("You must give a number of timesteps between [0, %d]\n", MAX_TIMESTEPS);
    }
  }

  void setBaseline(float baseline, int index = 0)
  {
    cost_baseline_and_norm_[index].x = baseline;
  };

  void setNormalizer(float normalizer, int index = 0)
  {
    cost_baseline_and_norm_[index].y = normalizer;
  };

  int getNumTimesteps() const
  {
    return this->params_.num_timesteps_;
  }

  /**
   * updates the scaling factor of noise for sampling around the nominal trajectory
   */
  void updateControlNoiseStdDev(const Eigen::Ref<const control_array>& sigma_u)
  {
    params_.control_std_dev_ = sigma_u;
    copyControlStdDevToDevice();
  }

  void disableFeedbackController()
  {
    enable_feedback_ = false;
  }

  void setFeedbackParams(TEMPLATED_FEEDBACK_PARAMS fb_params)
  {
    fb_controller_->setParams(fb_params);
  }

  bool getFeedbackEnabled() const
  {
    return enable_feedback_;
  }

  float getPercentageSampledControlTrajectories() const
  {
    return perc_sampled_control_trajectories_;
  }
  /**
   * Set the percentage of sample control trajectories to copy
   * back from the GPU. Multiplier is an integer in case the nominal
   * control trajectories also need to be saved.
   */
  void setPercentageSampledControlTrajectoriesHelper(float new_perc, int multiplier)
  {
    perc_sampled_control_trajectories_ = new_perc;
    sample_multiplier_ = multiplier;
    resizeSampledControlTrajectories(perc_sampled_control_trajectories_, sample_multiplier_,
                                     num_top_control_trajectories_);
  }

  void setTopNSampledControlTrajectoriesHelper(int new_top_num_samples)
  {
    num_top_control_trajectories_ = new_top_num_samples;
    resizeSampledControlTrajectories(perc_sampled_control_trajectories_, sample_multiplier_,
                                     num_top_control_trajectories_);
  }

  void resizeSampledControlTrajectories(float perc, int multiplier, int top_num);

  int getNumberSampledTrajectories() const
  {
    return perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  }

  int getNumberTopControlTrajectories() const
  {
    return num_top_control_trajectories_;
  }

  int getTotalSampledTrajectories() const
  {
    return getNumberSampledTrajectories() + getNumberTopControlTrajectories();
  }

  void setSlideControlScale(const Eigen::Ref<const control_array>& slide_control_scale)
  {
    params_.slide_control_scale_ = slide_control_scale;
  }

  /**
   * Return the most recent free energy calculation for the mean
   */
  MPPIFreeEnergyStatistics getFreeEnergyStatistics() const
  {
    return free_energy_statistics_;
  }

  std::vector<float> getSampledNoise();

  /**
   * Public data members
   */
  DYN_T* model_;
  COST_T* cost_;
  FB_T* fb_controller_;
  cudaStream_t stream_;
  cudaStream_t vis_stream_;

  float getDt() const
  {
    return params_.dt_;
  }
  void setDt(float dt)
  {
    params_.dt_ = dt;
    fb_controller_->setDt(dt);
  }

  float getLambda() const
  {
    return params_.lambda_;
  }
  void setLambda(float lambda)
  {
    params_.lambda_ = lambda;
  }

  float getAlpha() const
  {
    return params_.alpha_;
  }
  void setAlpha(float alpha)
  {
    params_.alpha_ = alpha;
  }

  PARAMS_T getParams() const
  {
    return params_;
  }

  void setParams(const PARAMS_T& p)
  {
    bool change_seed = p.seed_ != params_.seed_;
    bool change_std_dev = p.control_std_dev_ != params_.control_std_dev_;
    params_ = p;
    if (change_std_dev)
    {
      copyControlStdDevToDevice();
    }
    if (change_seed)
    {
      setSeedCUDARandomNumberGen(params_.seed_);
    }
  }

  int getNumIters() const
  {
    return params_.num_iters_;
  }
  void setNumIters(int num_iter)
  {
    params_.num_iters_ = num_iter;
  }

  float getDebug() const
  {
    return debug_;
  }
  void setDebug(bool debug)
  {
    debug_ = debug;
  }
  void setCUDAStream(cudaStream_t stream);

protected:
  // no default protected members
  void deallocateCUDAMemory();

  PARAMS_T params_;

  // TODO get raw pointers for different things
  bool debug_ = false;

  // Free energy variables
  MPPIFreeEnergyStatistics free_energy_statistics_;

  // float normalizer_;                             // Variable for the normalizing term from sampling.
  // float baseline_ = 0;                           // Baseline cost of the system.
  float perc_sampled_control_trajectories_ = 0;  // Percentage of sampled trajectories to return
  int sample_multiplier_ = 1;                    // How many nominal states we are keeping track of
  int num_top_control_trajectories_ = 0;         // Top n sampled trajectories to visualize
  std::vector<float> top_n_costs_;

  curandGenerator_t gen_;
  float* control_std_dev_d_;  // Array of size DYN_T::CONTROL_DIM
  float* initial_state_d_;    // Array of sizae DYN_T::STATE_DIM * (2 if there is a nominal state)

  Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2> control_history_;

  // one array of this size is allocated for each state we care about,
  // so it can be the size*N for N nominal states
  // [actual, nominal]
  float* control_d_;                  // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*N
  float* state_d_;                    // Array of size DYN_T::STATE_DIM*NUM_ROLLOUTS*N
  float* trajectory_costs_d_;         // Array of size NUM_ROLLOUTS*N
  float* control_noise_d_;            // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS*N
  float2* cost_baseline_and_norm_d_;  // Array of size number of systems
  control_trajectory control_ = control_trajectory::Zero();
  state_trajectory state_ = state_trajectory::Zero();
  sampled_cost_traj trajectory_costs_ = sampled_cost_traj::Zero();
  std::vector<float2> cost_baseline_and_norm_ = { make_float2(0.0, 0.0) };
  bool CUDA_mem_init_ = false;

  bool sampled_states_CUDA_mem_init_ = false;  // cudaMalloc, cudaFree boolean
  float* sampled_states_d_;                    // result of states that have been sampled from state trajectory kernel
  float* sampled_noise_d_;                     // noise to be passed to the state trajectory kernel
  float* sampled_costs_d_;       // result of cost that have been sampled from state and cost trajectory kernel
  int* sampled_crash_status_d_;  // result of crash_status that have been sampled
  std::vector<state_trajectory> sampled_trajectories_;  // sampled state trajectories from state trajectory kernel
  std::vector<cost_trajectory> sampled_costs_;
  std::vector<crash_status_trajectory> sampled_crash_status_;

  // Propagated real state trajectory
  state_trajectory propagated_feedback_state_trajectory_ = state_trajectory::Zero();

  // tracking controller variables
  bool enable_feedback_ = false;

  void copyControlStdDevToDevice(bool synchronize = true);

  void copyNominalControlToDevice(bool synchronize = true);

  /**
   * Saves the sampled controls from the GPU back to the CPU
   * Must be called after the rolloutKernel as that is when
   * du_d becomes the sampled controls
   */
  void copySampledControlFromDevice(bool synchronize = true);

  std::pair<int, float> findMinIndexAndValue(std::vector<int>& temp_list);
  void copyTopControlFromDevice(bool synchronize = true);

  void createAndSeedCUDARandomNumberGen();

  void setSeedCUDARandomNumberGen(unsigned seed);

  /**
   * Allocates CUDA memory for actual states and nominal states if needed
   * @param nominal_size if only actual this should be 0
   */
  void allocateCUDAMemoryHelper(int nominal_size = 0, bool allocate_double_noise = true);

  // TODO all the copy to device functions to streamline process
private:
  // ======== MUST BE OVERWRITTEN =========
  void allocateCUDAMemory()
  {
    allocateCUDAMemoryHelper();
  };
  /**
   * TODO all copy to device and back functions implemented for specific controller
   * When you write your own you must control when synchronize stream is called
   */
  // ======== END MUST BE OVERWRITTEN =====
};

#ifdef __CUDACC__
#include "controller.cu"
#endif

#endif  // MPPIGENERIC_CONTROLLER_CUH
