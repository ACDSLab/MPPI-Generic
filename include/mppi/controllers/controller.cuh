//
// Created by jason on 10/30/19.
//

#ifndef MPPIGENERIC_CONTROLLER_CUH
#define MPPIGENERIC_CONTROLLER_CUH

#include <array>
#include <Eigen/Core>
#include <chrono>
#include <mppi/ddp/util.h>
#include "curand.h"

#include <mppi/core/mppi_common.cuh>

#include <mppi/ddp/ddp_model_wrapper.h>
#include <mppi/ddp/ddp_tracking_costs.h>
#include <mppi/ddp/ddp.h>
#include <mppi/utils/math_utils.h>

#include <cfloat>

struct freeEnergyEstimate {
  float increase = -1;
  float previousBaseline = -1;
  float freeEnergyMean = -1;
  float freeEnergyVariance = -1;
  float freeEnergyModifiedVariance = -1;
  float normalizerPercent = -1;
};

struct MPPIFreeEnergyStatistics {
  int nominal_state_used = 0;

  freeEnergyEstimate nominal_sys;
  freeEnergyEstimate real_sys;
};

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
class Controller {
public:
  //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * typedefs for access to templated class from outside classes
   */
  typedef DYN_T TEMPLATED_DYNAMICS;
  typedef COST_T TEMPLATED_COSTS;
  // MAX_TIMESTEPS is defined as an upper bound, if lower that region is just ignored when calculating control
  // does not reallocate cuda memory
  int num_timesteps_ = MAX_TIMESTEPS;

  /**
   * Aliases
   */
   // Control typedefs
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, MAX_TIMESTEPS> control_trajectory; // A control trajectory
//  typedef util::NamedEigenAlignedVector<control_trajectory> sampled_control_traj;
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;

  // State typedefs
  using state_array = typename DYN_T::state_array;
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM, MAX_TIMESTEPS> state_trajectory; // A state trajectory
//  typedef util::NamedEigenAlignedVector<state_trajectory> sampled_state_traj;

  // Cost typedefs
  typedef Eigen::Matrix<float, MAX_TIMESTEPS, 1> cost_trajectory;
  typedef Eigen::Matrix<float, NUM_ROLLOUTS, 1> sampled_cost_traj;
//  typedef std::array<float, MAX_TIMESTEPS> cost_trajectory; // A cost trajectory
//  typedef std::array<float, NUM_ROLLOUTS> sampled_cost_traj; // All costs sampled for all rollouts

  // tracking controller typedefs
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

  Controller(DYN_T* model, COST_T* cost, float dt, int max_iter,
          float lambda, float alpha,
          const Eigen::Ref<const control_array>& control_std_dev,
          int num_timesteps = MAX_TIMESTEPS,
          const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
          cudaStream_t stream = nullptr) {
    model_ = model;
    cost_ = cost;
    dt_ = dt;
    num_iters_ = max_iter;
    lambda_ = lambda;
    alpha_ = alpha;
    num_timesteps_ = num_timesteps;

    control_std_dev_ = control_std_dev;
    control_ = init_control_traj;
    control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

    // Create the random number generator
    createAndSeedCUDARandomNumberGen();

    // Bind the model and control to the given stream
    setCUDAStream(stream);

    // Call the GPU setup functions of the model and cost
    model_->GPUSetup();
    cost_->GPUSetup();

    // allocate memory for the optimizer result
    result_ = OptimizerResult<ModelWrapperDDP<DYN_T>>();
    result_.feedback_gain = feedback_gain_trajectory(MAX_TIMESTEPS);
    for(int i = 0; i < MAX_TIMESTEPS; i++) {
      result_.feedback_gain[i] = Eigen::Matrix<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM>::Zero();
    }

    /**
     * When implementing your own version make sure to write your own allocateCUDAMemroy and call it from the constructor
     * along with any other methods to copy memory to the device and back
     */
     // TODO pass function pointer?
  }

  // TODO should be private with test as a friend to ensure it is only used in testing
  Controller() = default;

  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a basePlant reference
   */
  virtual ~Controller() {
    // Free the CUDA memory of every object
    model_->freeCudaMem();
    cost_->freeCudaMem();
    cudaFree(sampled_noise_d_);
    cudaFree(sampled_states_d_);

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

  virtual std::string getControllerName() {return "name not set";};
  virtual std::string getCostFunctionName() {return cost_->getCostFunctionName();}


  virtual std::vector<state_trajectory> getSampledStateTrajectories() {
    return sampled_trajectories_;
  }

  /**
   * only used in rmppi, here for generic calls in base_plant. Jank as hell
   * @param state
   * @param stride
   */
  void updateImportanceSamplingControl(const Eigen::Ref<const state_array> &state, int optimization_stride) {}

  /**
   * Used to update the importance sampler
   * @param nominal_control the new nominal control sequence to sample around
   */
  virtual void updateImportanceSampler(const Eigen::Ref<const control_trajectory>& nominal_control) {
    // TODO copy to device new control sequence
    control_ = nominal_control;
  }

  /**
   * determines the control that should
   * @param state
   * @param rel_time
   * @return
   */
  virtual control_array getCurrentControl(state_array& state, double rel_time,
          state_array& target_nominal_state, control_trajectory& c_traj, feedback_gain_trajectory& gain_traj) {
    // MPPI control
    control_array u_ff = interpolateControls(rel_time, c_traj);
    control_array u_fb = control_array::Zero();
    if(enable_feedback_) {
       u_fb = interpolateFeedback(state, target_nominal_state, gain_traj, rel_time);
    }
    control_array result = u_ff + u_fb;
    //printf("rel_time %f\n", rel_time);
    //printf("uff: %f, %f u_fb: %f, %f\n", u_ff[0], u_ff[1], u_fb[0], u_fb[1]);

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
  virtual control_array interpolateControls(double rel_time, control_trajectory& c_traj) {
    int lower_idx = (int) (rel_time / dt_);
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * dt_) / dt_;

    control_array interpolated_control;
    control_array prev_cmd = c_traj.col(lower_idx);
    control_array next_cmd = c_traj.col(upper_idx);
    interpolated_control = (1 - alpha) * prev_cmd + alpha * next_cmd;

    //printf("prev: %d %f, %f\n", lower_idx, prev_cmd[0], prev_cmd[1]);
    //printf("next: %d %f, %f\n", upper_idx, next_cmd[0], next_cmd[1]);
    //printf("smoother: %f\n", alpha);
    return interpolated_control;
  }

  virtual state_array interpolateState(state_trajectory& s_traj, double rel_time) {
    int lower_idx = (int) (rel_time / dt_);
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * dt_) / dt_;

    state_array desired_state;
    desired_state = (1 - alpha)*s_traj.col(lower_idx) + alpha*s_traj.col(upper_idx);
    return desired_state;
  }

  /**
   *
   * @param state
   * @param rel_time
   * @return
   */
  virtual control_array interpolateFeedback(state_array& state, state_array& target_nominal_state,
          feedback_gain_trajectory& gain_traj, double rel_time) {
    int lower_idx = (int) (rel_time / dt_);
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * dt_) / dt_;

    control_array u_fb = ((1-alpha)*gain_traj[lower_idx]
            + alpha*gain_traj[upper_idx])*(state - target_nominal_state);

    return u_fb;
  }

  /**
   * returns the current control sequence
   */
  virtual control_trajectory getControlSeq() {
    return control_;
  };

  /**
   * Gets the state sequence of the nominal trajectory
   */
  virtual state_trajectory getStateSeq() {
    return state_;
  }

  /**
   * Return all the sampled costs sequences
   */
  virtual sampled_cost_traj getSampledCostSeq() {
    return trajectory_costs_;
  };

  /**
   * Return control feedback gains
   */
  virtual feedback_gain_trajectory getFeedbackGains() {
    if(enable_feedback_) {
      return result_.feedback_gain;
    } else {
      return feedback_gain_trajectory();
    }
  };

  // Indicator for algorithm health, should be between 0.01 and 0.1 anecdotally
  float getNormalizerPercent() {return this->normalizer_/(float)NUM_ROLLOUTS;}

  /**
 * Computes the actual trajectory given the MPPI optimal control and the
 * feedback gains computed by DDP. If feedback is not enabled, then we return
 * zero since this function would not make sense.
 */
  virtual void computeFeedbackPropagatedStateSeq() {
    if (!enable_feedback_) {
      return;
    }
    // Compute the nominal trajectory
    propagated_feedback_state_trajectory_.col(0) = getAncillaryStateSeq().col(0); // State that we optimized from
    state_array xdot;
    state_array current_state;
    control_array current_control;
    for (int i =0; i < num_timesteps_ - 1; ++i) {
      current_state = propagated_feedback_state_trajectory_.col(i);
      // MPPI control apply feedback at the given timestep against the nominal trajectory at that timestep
      current_control = getControlSeq().col(i) + getFeedbackGains()[i]*(current_state - getStateSeq().col(i));
      model_->computeStateDeriv(current_state, current_control, xdot);
      model_->updateState(current_state, xdot, dt_);
      propagated_feedback_state_trajectory_.col(i+1) = current_state;
    }
  }

  /**
   *
   * @return State trajectory from optimized state with MPPI control and computed feedback gains
   */
  state_trajectory getFeedbackPropagatedStateSeq() {return propagated_feedback_state_trajectory_;};

  control_array getControlStdDev() { return control_std_dev_;};

  float getBaselineCost() {return baseline_;};
  float getNormalizerCost() {return normalizer_;};

  // TODO is this what we want?
  state_trajectory getAncillaryStateSeq() {return result_.state_trajectory;};

  virtual void initDDP(const StateCostWeight& q_mat,
                       const Hessian& q_f_mat,
                       const ControlCostWeight& r_mat) {
    enable_feedback_ = true;

    util::DefaultLogger logger;
    bool verbose = false;
    ddp_model_  = std::make_shared<ModelWrapperDDP<DYN_T>>(model_);
    ddp_solver_ = std::make_shared< DDP<ModelWrapperDDP<DYN_T>>>(dt_,
            num_timesteps_, 1, &logger, verbose);
    Q_ = q_mat;
    Qf_ = q_f_mat;
    R_ = r_mat;

    for (int i = 0; i < DYN_T::CONTROL_DIM; i++) {
      control_min_(i) = model_->control_rngs_[i].x;
      control_max_(i) = model_->control_rngs_[i].y;
    }

    run_cost_ = std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(Q_,
            R_, num_timesteps_);
    terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(Qf_);
  }

  virtual void computeFeedbackGains(const Eigen::Ref<const state_array>& state) {
    if(!enable_feedback_) {
      return;
    }

    run_cost_->setTargets(getStateSeq().data(), getControlSeq().data(),
                          num_timesteps_);

    terminal_cost_->xf = run_cost_->traj_target_x_.col(num_timesteps_ - 1);
    result_ = ddp_solver_->run(state, control_,
                               *ddp_model_, *run_cost_, *terminal_cost_,
                               control_min_, control_max_);
  }

  void smoothControlTrajectoryHelper(Eigen::Ref<control_trajectory> u, const Eigen::Ref<Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>>& control_history) {
    // TODO generalize to any size filter
    // TODO does the logic of handling control history reasonable?

    // Create the filter coefficients
    Eigen::Matrix<float, 1, 5> filter_coefficients;
    filter_coefficients << -3, 12, 17, 12, -3;
    filter_coefficients /= 35.0;

    // Create and fill a control buffer that we can apply the convolution filter
    Eigen::Matrix<float, MAX_TIMESTEPS+4, DYN_T::CONTROL_DIM> control_buffer;

    // Fill the first two timesteps with the control history
    control_buffer.topRows(2) = control_history.transpose();

    // Fill the center timesteps with the current nominal trajectory
    control_buffer.middleRows(2, MAX_TIMESTEPS) = u.transpose();

    // Fill the last two timesteps with the end of the current nominal control trajectory
    control_buffer.row(MAX_TIMESTEPS+2) = u.transpose().row(MAX_TIMESTEPS-1);
    control_buffer.row(MAX_TIMESTEPS+3) = u.transpose().row(MAX_TIMESTEPS-1);

    // Apply convolutional filter to each timestep
    for (int i = 0; i < MAX_TIMESTEPS; ++i) {
      u.col(i) = (filter_coefficients*control_buffer.middleRows(i,5)).transpose();
    }
  }

  virtual void slideControlSequenceHelper(int steps, Eigen::Ref<control_trajectory> u) {
    for (int i = 0; i < num_timesteps_; ++i) {
      int ind = std::min(i + steps, num_timesteps_ - 1);
      u.col(i) = u.col(ind);
      if (i + steps > num_timesteps_ - 1) {
        u.col(i) = model_->zero_control_;
      }
    }
  }

  virtual void saveControlHistoryHelper(int steps,
          const Eigen::Ref<const control_trajectory>& u_trajectory,
          Eigen::Ref<Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>> u_history) {
    if (steps == 1) { // We only moved one timestep
      u_history.col(0) = u_history.col(1);
      u_history.col(1) = u_trajectory.col(0);
    }
    else if (steps >= 2) { // We have moved more than one timestep, but our history size is still only 2
      u_history.col(0) = u_trajectory.col(steps - 2);
      u_history.col(1) = u_trajectory.col(steps - 1);
    }
  }

  /**
   * Reset Controls
   */
  virtual void resetControls() {
    // TODO
  };

  virtual void computeStateTrajectoryHelper(Eigen::Ref<state_trajectory> result, const Eigen::Ref<const state_array>& x0,
          const Eigen::Ref<const control_trajectory>& u) {
    result.col(0) = x0;
    state_array xdot;
    state_array state;
    for (int i =0; i < num_timesteps_ - 1; ++i) {
      state = result.col(i);
      model_->computeStateDeriv(state, u.col(i), xdot);
      model_->updateState(state, xdot, dt_);
      result.col(i+1) = state;
    }
  }

  void setNumTimesteps(int num_timesteps) {
    // TODO fix the tracking controller as well
    if ((num_timesteps <= MAX_TIMESTEPS) && (num_timesteps > 0)) {
      num_timesteps_ = num_timesteps;
    } else {
      num_timesteps_ = MAX_TIMESTEPS;
      printf("You must give a number of timesteps between [0, %d]\n", MAX_TIMESTEPS);
    }
  }
  int getNumTimesteps() {return num_timesteps_;}

  /**
   * updates the scaling factor of noise for sampling around the nominal trajectory
   */
  void updateControlNoiseStdDev(const Eigen::Ref<const control_array>& sigma_u) {
    //std::cout << control_std_dev_ << std::endl;
    control_std_dev_ = sigma_u;
    //std::cout << control_std_dev_ << std::endl;
    copyControlStdDevToDevice();
  }

  void setFeedbackController(bool enable_feedback) {
    enable_feedback_ = enable_feedback;
  }
  bool getFeedbackEnabled() {return enable_feedback_;}

  /**
   * Set the percentage of sample control trajectories to copy
   * back from the GPU. Multipler is an integer in case the nominal
   * control trajectories also need to be saved.
   */
  void setPercentageSampledControlTrajectoriesHelper(float new_perc, int multiplier) {
    int num_sampled_trajectories = new_perc * NUM_ROLLOUTS;

    HANDLE_ERROR(cudaMalloc((void**)&sampled_states_d_,
                            sizeof(float)*DYN_T::STATE_DIM*num_timesteps_*num_sampled_trajectories*multiplier));
    HANDLE_ERROR(cudaMalloc((void**)&sampled_noise_d_,
                            sizeof(float)*DYN_T::CONTROL_DIM*num_timesteps_*num_sampled_trajectories*multiplier));

    sampled_trajectories_.resize(num_sampled_trajectories*multiplier);
    perc_sampled_control_trajectories = new_perc;
  }

  int getNumberSampledTrajectories() {
    return perc_sampled_control_trajectories * NUM_ROLLOUTS;
  }

  /**
   * Return a percentage of sampled control trajectories from the latest rollout
   */
  std::vector<control_trajectory> getSampledControlSeq() {return sampled_controls_;}

  /**
   * Return the most recent free energy calculation for the mean
   */
   MPPIFreeEnergyStatistics getFreeEnergyStatistics() {return free_energy_statistics_;}

  std::vector<float> getSampledNoise() {
    std::vector<float> vector = std::vector<float>(NUM_ROLLOUTS*num_timesteps_*DYN_T::CONTROL_DIM, FLT_MIN);

    HANDLE_ERROR(cudaMemcpyAsync(vector.data(), control_noise_d_, sizeof(float)*NUM_ROLLOUTS*num_timesteps_*DYN_T::CONTROL_DIM,
                                 cudaMemcpyDeviceToHost, stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
    return vector;
  }

  /**
   * Public data members
   */
  DYN_T* model_;
  COST_T* cost_;
  cudaStream_t stream_;

  float getDt() {return dt_;}
  void setDt(float dt) {dt_ = dt;}

  float getDebug() {return debug_;}
  void setDebug(float debug) {debug_ = debug;}

protected:
  // no default protected members
  void deallocateCUDAMemory();

  // TODO get raw pointers for different things
  bool debug_ = false;

  // Free energy variables
  MPPIFreeEnergyStatistics free_energy_statistics_;

  int num_iters_;  // Number of optimization iterations
  float dt_;
  float lambda_; // Value of the temperature in the softmax.
  float alpha_; //

  float normalizer_; // Variable for the normalizing term from sampling.
  float baseline_ = 0; // Baseline cost of the system.
  float perc_sampled_control_trajectories = 0; // Percentage of sampled trajectories to return

  curandGenerator_t gen_;
  control_array control_std_dev_ = control_array::Zero();
  float* control_std_dev_d_; // Array of size DYN_T::CONTROL_DIM
  float* initial_state_d_; // Array of sizae DYN_T::STATE_DIM * (2 if there is a nominal state)

  Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2> control_history_;

  // one array of this size is allocated for each state we care about,
  // so it can be the size*N for N nominal states
  // [actual, nominal]
  float* control_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*N
  float* state_d_; // Array of size DYN_T::STATE_DIM*NUM_ROLLOUTS*N
  float* trajectory_costs_d_; // Array of size NUM_ROLLOUTS*N
  float* control_noise_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS*N
  control_trajectory control_ = control_trajectory::Zero();
  state_trajectory state_ = state_trajectory::Zero();
  sampled_cost_traj trajectory_costs_ = sampled_cost_traj::Zero();


  float* sampled_states_d_; // result of states that have been sampled from state trajectory kernel
  float* sampled_noise_d_; // noise to be passed to the state trajectory kernel
  std::vector<control_trajectory> sampled_controls_; // Sampled control trajectories from rollout kernel
  std::vector<state_trajectory> sampled_trajectories_; // sampled state trajectories from state trajectory kernel

  // Propagated real state trajectory
  state_trajectory propagated_feedback_state_trajectory_ = state_trajectory::Zero();

  // tracking controller variables
  StateCostWeight Q_;
  Hessian Qf_;
  ControlCostWeight R_;
  bool enable_feedback_ = false;

  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;

  // for DDP
  control_array control_min_;
  control_array control_max_;

  OptimizerResult<ModelWrapperDDP<DYN_T>> result_;

  void copyControlStdDevToDevice();

  void copyNominalControlToDevice();

  /**
   * Saves the sampled controls from the GPU back to the CPU
   * Must be called after the rolloutKernel as that is when
   * du_d becomes the sampled controls
   */
  void copySampledControlFromDevice();

  void setCUDAStream(cudaStream_t stream);

  void createAndSeedCUDARandomNumberGen();

  /**
   * Allocates CUDA memory for actual states and nominal states if needed
   * @param nominal_size if only actual this should be 0
   */
  void allocateCUDAMemoryHelper(int nominal_size = 0,
                                bool allocate_double_noise = true);

  // TODO all the copy to device functions to streamline process
private:
  // ======== MUST BE OVERWRITTEN =========
  void allocateCUDAMemory() {
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

#endif //MPPIGENERIC_CONTROLLER_CUH
