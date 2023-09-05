/**
 * Created by Bogdan on 2/11/20.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
 **/

#ifndef BASE_PLANT_H_
#define BASE_PLANT_H_

// Double check if these are included in mppi_common.h
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>
#include <mppi/controllers/controller.cuh>
#include <mppi/utils/math_utils.h>

template <class CONTROLLER_T>
class BasePlant
{
public:
  using c_array = typename CONTROLLER_T::control_array;
  using c_traj = typename CONTROLLER_T::control_trajectory;

  using s_array = typename CONTROLLER_T::state_array;
  using s_traj = typename CONTROLLER_T::state_trajectory;
  // using K_traj = typename CONTROLLER_T::feedback_gain_trajectory;

  using o_array = typename CONTROLLER_T::output_array;
  using o_traj = typename CONTROLLER_T::output_trajectory;

  using DYN_T = typename CONTROLLER_T::TEMPLATED_DYNAMICS;
  using DYN_PARAMS_T = typename DYN_T::DYN_PARAMS_T;
  using COST_T = typename CONTROLLER_T::TEMPLATED_COSTS;
  using COST_PARAMS_T = typename COST_T::COST_PARAMS_T;
  using TEMPLATED_CONTROLLER = CONTROLLER_T;
  using CONTROLLER_PARAMS_T = typename CONTROLLER_T::TEMPLATED_PARAMS;

  // Feedback related aliases
  using FB_STATE_T = typename CONTROLLER_T::TEMPLATED_FEEDBACK::TEMPLATED_FEEDBACK_STATE;

  typedef std::map<std::string, Eigen::VectorXf> buffer_trajectory;

protected:
  std::mutex access_guard_;

  int hz_ = 10;  // Frequency of control publisher
  int visualization_hz_ = 5;
  bool debug_mode_ = false;

  DYN_PARAMS_T dynamics_params_;
  std::mutex dynamics_params_guard_;
  COST_PARAMS_T cost_params_;
  std::mutex cost_params_guard_;
  CONTROLLER_PARAMS_T controller_params_;
  std::mutex controller_params_guard_;

  std::atomic<bool> has_new_dynamics_params_{ false };
  std::atomic<bool> has_new_cost_params_{ false };
  std::atomic<bool> has_new_controller_params_{ false };
  std::atomic<bool> enabled_{ false };

  // Values needed
  s_array init_state_ = s_array::Zero();
  c_array init_u_ = c_array::Zero();

  // Values updated at every time step
  s_array state_ = s_array::Zero();
  c_array u_ = c_array::Zero();
  // solution
  s_traj state_traj_;
  c_traj control_traj_;
  o_traj output_traj_;

  // values sometime updated
  // TODO init to zero?
  FB_STATE_T feedback_state_;

  // from ROSHandle mppi_node
  int optimization_stride_ = 1;
  int last_optimization_stride_ = 0;

  /**
   * From before while loop
   */
  /**
   * Robot Time: based off of the time stamps from the state estimator
   * Wall Clock: always real time per the computer
   */
  // Robot Time: can scale with a simulation
  std::atomic<double> last_used_state_update_time_{ -1.0 };  // time of the last state update that was used for
                                                             // optimization
  std::atomic<double> state_time_{ -1.0 };

  std::atomic<bool> use_real_time_timing_{ false };
  // Wall Clock: always real time
  double optimize_loop_duration_ = 0;  // duration of the entire controller run loop
  double optimization_duration_ = 0;   // Most recent time it took to run MPPI iteration
  double feedback_duration_ = 0;       // most recent time it took to run the feedback controller
  double sleep_duration_ = 0;          // how long the most recent loop in runControlLoop slept
  double avg_loop_time_ms_ = 0;        // Average time it takes to run the controller
  double avg_optimize_time_ms_ = 0;    // Average time it takes to runControlLoop
  double avg_feedback_time_ms_ = 0;    // Average time it takes to run the feedback controller
  double avg_sleep_time_ms_ = 0;       // Average time the runControlLoop sleeps between calls

  int num_iter_ = 0;  // number of calls to computeControl
  /**
   * represents the status of the vehicle
   * 0: running normally
   * 1: not activated or no state information
   */
  int status_ = 1;

public:
  std::shared_ptr<CONTROLLER_T> controller_;

  //  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BasePlant(std::shared_ptr<CONTROLLER_T> controller, int hz, int optimization_stride)
  {
    controller_ = controller;
    hz_ = hz;
    optimization_stride_ = optimization_stride;
    control_traj_ = c_traj::Zero();
    state_traj_ = s_traj::Zero();
    dynamics_params_ = controller->model_->getParams();
    cost_params_ = controller_->cost_->getParams();
  };
  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a BasePlant reference
   */
  virtual ~BasePlant() = default;

  // ======== PURE VIRTUAL =========
  /**
   * applies the control to the system
   * @param u
   */
  virtual void pubControl(const c_array& u) = 0;

  /**
   * publishes the target nominal state
   * @param s
   */
  virtual void pubNominalState(const s_array& s) = 0;

  virtual void pubFreeEnergyStatistics(MPPIFreeEnergyStatistics& fe_stats) = 0;

  /**
   * @brief Checks the system status.
   * @return An integer specifying the status. 0 means the system is operating
   * nominally, 1 means something is wrong but no action needs to be taken,
   * 2 means that the vehicle should stop immediately.
   */
  virtual int checkStatus() = 0;

  /**
   * @return the current time not necessarily real time
   */
  virtual double getCurrentTime() = 0;

  /**
   * @return the timestamp from the most recent position
   */
  virtual double getPoseTime() = 0;

  /**
   * gets the correct time for the state message
   * @return
   */
  virtual double getStateTime()
  {
    return getPoseTime();
  }

  void setUseRealTimeTiming(bool use_real_time)
  {
    use_real_time_timing_ = use_real_time;
  }

  // ======== PURE VIRTUAL END ====

  s_traj getStateTraj()
  {
    return state_traj_;
  }
  c_traj getControlTraj()
  {
    return control_traj_;
  }
  FB_STATE_T getFeedbackState()
  {
    return feedback_state_;
  }

  /**
   * Return the latest state received
   * @return the latest state
   */
  virtual s_array getState()
  {
    std::lock_guard<std::mutex> lck(access_guard_);
    return state_;
  };

  virtual void setState(s_array state)
  {
    state_ = state;
  }
  virtual void setControl(c_array u)
  {
    u_ = u;
  }
  virtual void setDebugMode(bool mode)
  {
    debug_mode_ = mode;
  }

  void resetStateTime()
  {
    last_used_state_update_time_ = -1;
  };

  double getAvgOptimizationTime() const
  {
    return avg_optimize_time_ms_;
  };

  int getTargetOptimizationStride()
  {
    return optimization_stride_;
  };
  int getLastOptimizationStride()
  {
    return last_optimization_stride_;
  };
  void setTargetOptimizationStride(int new_val)
  {
    optimization_stride_ = new_val;
  }

  int getHz() const
  {
    return hz_;
  }

  void setHz(int hz)
  {
    hz_ = hz;
  }

  int getVisualizationHz() const
  {
    return visualization_hz_;
  }

  void setVisualizationHz(int hz)
  {
    visualization_hz_ = hz;
  }

  virtual buffer_trajectory getSmoothedBuffer(double time)
  {
    throw std::logic_error("Invalid dynamics with current plant, it requires the buffered plant");
  }

  virtual void setSolution(const s_traj& state_seq, const c_traj& control_seq, const o_traj& output_seq,
                           const FB_STATE_T& fb_state, double timestamp)
  {
    last_used_state_update_time_ = timestamp;
    std::lock_guard<std::mutex> guard(access_guard_);
    state_traj_ = state_seq;
    output_traj_ = output_seq;
    control_traj_ = control_seq;
    feedback_state_ = fb_state;
    num_iter_++;
  }

  /**
   * updates the state and publishes a new control
   * @param state the most recent state from state estimator
   * @param time the time of the most recent state from the state estimator
   */
  virtual void updateState(s_array& state, double time)
  {
    // calculate and update all timing variables
    double temp_last_state_update_time = last_used_state_update_time_;

    double time_since_last_opt = time - temp_last_state_update_time;

    state_ = state;
    state_time_ = time;

    // check if the requested time is in the calculated trajectory
    bool t_within_trajectory =
        time >= temp_last_state_update_time &&
        time < temp_last_state_update_time + controller_->getDt() * controller_->getNumTimesteps();

    // TODO check that we haven't been waiting too long
    if (time_since_last_opt > 0 && t_within_trajectory)
    {
      s_array target_nominal_state = this->controller_->interpolateState(state_traj_, time_since_last_opt);
      pubControl(controller_->getCurrentControl(state, time_since_last_opt, target_nominal_state, control_traj_,
                                                feedback_state_));
      if (debug_mode_)
      {
        pubNominalState(target_nominal_state);
      }
    }
  }

  virtual bool hasNewDynamicsParams()
  {
    return has_new_dynamics_params_;
  };
  virtual bool hasNewCostParams()
  {
    return has_new_cost_params_;
  };
  virtual bool hasNewControllerParams()
  {
    return has_new_controller_params_;
  };

  virtual DYN_PARAMS_T getNewDynamicsParams(bool set_flag = false)
  {
    has_new_dynamics_params_ = set_flag;
    return dynamics_params_;
  }
  virtual COST_PARAMS_T getNewCostParams(bool set_flag = false)
  {
    has_new_cost_params_ = set_flag;
    return cost_params_;
  }
  virtual CONTROLLER_PARAMS_T getNewControllerParams(bool set_flag = false)
  {
    has_new_controller_params_ = set_flag;
    return controller_params_;
  }

  virtual void setDynamicsParams(DYN_PARAMS_T params)
  {
    std::lock_guard<std::mutex> guard(dynamics_params_guard_);
    dynamics_params_ = params;
    has_new_dynamics_params_ = true;
  }
  virtual void setCostParams(COST_PARAMS_T params)
  {
    std::lock_guard<std::mutex> guard(cost_params_guard_);
    cost_params_ = params;
    has_new_cost_params_ = true;
  }
  virtual void setControllerParams(CONTROLLER_PARAMS_T params)
  {
    std::lock_guard<std::mutex> guard(controller_params_guard_);
    controller_params_ = params;
    has_new_controller_params_ = true;
  }

  /**
   *
   * @param controller
   * @param state
   * @return
   */
  virtual bool updateParameters()
  {
    bool changed = false;
    // Update cost parameters
    if (hasNewCostParams())
    {
      std::lock_guard<std::mutex> guard(cost_params_guard_);
      changed = true;
      COST_PARAMS_T cost_params = getNewCostParams();
      controller_->cost_->setParams(cost_params);
    }
    // Update dynamics params
    if (hasNewDynamicsParams())
    {
      std::lock_guard<std::mutex> guard(dynamics_params_guard_);
      changed = true;
      DYN_PARAMS_T dyn_params = getNewDynamicsParams();
      controller_->model_->setParams(dyn_params);
    }
    // Update controller params
    if (hasNewControllerParams())
    {
      std::lock_guard<std::mutex> guard(controller_params_guard_);
      changed = true;
      CONTROLLER_PARAMS_T controller_params = getNewControllerParams();
      controller_->setParams(controller_params);
    }
    return changed;
  }

  /**
   * two concepts of time
   *    1. wall clock: how long it takes according to actual time to optimize
   *    2. robot time: how long has elapsed from the perspective of the robot (per the state estimator)
   * @param controller
   * @param is_alive
   * @return the millisecond number that the loop iteration started at
   */
  void runControlIteration(std::atomic<bool>* is_alive)
  {
    std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
    if (!is_alive->load())
    {
      // break out if it should stop
      return;
    }

    double temp_last_state_time = getStateTime();
    double temp_last_used_state_update_time = last_used_state_update_time_;

    // wait for a new state to compute control sequence from
    int counter = 0;
    while (temp_last_used_state_update_time == temp_last_state_time && is_alive->load())
    {
      usleep(50);
      temp_last_state_time = getStateTime();
      counter++;
    }
    // TODO could this cause misalignment
    this->access_guard_.lock();
    s_array state = state_;
    temp_last_state_time = state_time_;
    this->access_guard_.unlock();

    if (this->controller_->model_->checkRequiresBuffer())
    {
      std::lock_guard<std::mutex> guard(dynamics_params_guard_);
      this->controller_->model_->updateFromBuffer(this->getSmoothedBuffer(temp_last_state_time));
      HANDLE_ERROR(cudaStreamSynchronize(this->controller_->model_->stream_));
    }

    // Check the robots status for this optimization
    int temp_status = checkStatus();

    updateParameters();

    // calculate how much we should slide the control sequence
    double dt = temp_last_state_time - temp_last_used_state_update_time;
    if (temp_last_used_state_update_time == -1)
    {  //
      // should only happen on the first iteration
      dt = 0;
      last_optimization_stride_ = 0;
    }
    else
    {
      last_optimization_stride_ = std::max(int(round(dt / this->controller_->getDt())), optimization_stride_);
    }
    // printf("calc optimization stride %f %f %f %d\n", dt, temp_last_used_state_update_time, temp_last_state_time,
    // last_optimization_stride_);
    // determine how long we should stride based off of robot time

    if (last_optimization_stride_ > 0 && last_optimization_stride_ < controller_->getNumTimesteps())
    {
      controller_->updateImportanceSamplingControl(state, last_optimization_stride_);
      controller_->slideControlSequence(last_optimization_stride_);
    }

    // Compute a new control sequence
    std::chrono::steady_clock::time_point optimization_start = std::chrono::steady_clock::now();
    controller_->computeControl(state, last_optimization_stride_);  // Compute the nominal control sequence

    MPPIFreeEnergyStatistics fe_stats = controller_->getFreeEnergyStatistics();

    c_traj control_traj = controller_->getControlSeq();
    if (!control_traj.allFinite())
    {
      std::cerr << "ERROR: Nan in control inside plant" << std::endl;
      std::cerr << control_traj << std::endl;
      exit(-1);
    }
    s_traj state_traj = controller_->getTargetStateSeq();
    if (!state_traj.allFinite())
    {
      std::cerr << "ERROR: Nan in state inside plant" << std::endl;
      std::cerr << state_traj << std::endl;
      exit(-1);
    }
    o_traj output_traj = controller_->getTargetOutputSeq();
    if (!state_traj.allFinite())
    {
      std::cerr << "ERROR: Nan in state inside plant" << std::endl;
      std::cerr << state_traj << std::endl;
      exit(-1);
    }
    optimization_duration_ = mppi::math::timeDiffms(std::chrono::steady_clock::now(), optimization_start);

    std::chrono::steady_clock::time_point feedback_start = std::chrono::steady_clock::now();
    // TODO make sure this is zero by default
    FB_STATE_T feedback_state;
    if (controller_->getFeedbackEnabled())
    {
      controller_->computeFeedback(state);
      feedback_state = controller_->getFeedbackState();
    }
    feedback_duration_ = mppi::math::timeDiffms(std::chrono::steady_clock::now(), feedback_start);

    // Set the updated solution for execution
    setSolution(state_traj, control_traj, output_traj, feedback_state, temp_last_state_time);
    status_ = temp_status;
    pubFreeEnergyStatistics(fe_stats);

    // calculate the propogated feedback trajectory
    controller_->computeFeedbackPropagatedStateSeq();

    // Update the average loop time data
    double prev_iter_percent = (num_iter_ - 1.0) / num_iter_;

    avg_optimize_time_ms_ = prev_iter_percent * avg_optimize_time_ms_ + optimization_duration_ / num_iter_;
    avg_feedback_time_ms_ = prev_iter_percent * avg_feedback_time_ms_ + feedback_duration_ / num_iter_;

    optimize_loop_duration_ = mppi::math::timeDiffms(std::chrono::steady_clock::now(), loop_start);
    avg_loop_time_ms_ = prev_iter_percent * avg_loop_time_ms_ + optimize_loop_duration_ / num_iter_;
  }

  void runControlLoop(std::atomic<bool>* is_alive)
  {
    // Initial condition of the robot
    state_ = init_state_;

    // Initial control value
    u_ = init_u_;

    controller_->resetControls();

    // Start the control loop.
    while (is_alive->load())
    {
      double wait_until_real_time = getCurrentTime() + (1.0 / hz_) * optimization_stride_;
      runControlIteration(is_alive);

      double wait_until_state_time = last_used_state_update_time_ + (1.0 / hz_) * optimization_stride_;
      // printf("last used state update time %f last_stride = %d\n", last_used_state_update_time_,
      // last_optimization_stride_); printf("wait until time %f current time %f\n", wait_until_time, getCurrentTime());

      std::chrono::steady_clock::time_point sleep_start = std::chrono::steady_clock::now();
      while (is_alive->load() && (wait_until_state_time > getStateTime() ||
                                  (wait_until_real_time > getCurrentTime() && use_real_time_timing_)))
      {
        updateParameters();
        usleep(50);
      }
      sleep_duration_ = mppi::math::timeDiffms(std::chrono::steady_clock::now(), sleep_start);
      double prev_iter_percent = (num_iter_ - 1.0) / num_iter_;
      avg_sleep_time_ms_ = prev_iter_percent * avg_sleep_time_ms_ + sleep_duration_ / num_iter_;
      // printf("sleep: %f loop_time %f at time %f", sleep_duration_, optimize_loop_duration_, getCurrentTime());
      // std::cout << " loop ended at: " <<
      // std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
      // << std::endl;
    }
  }
};

#endif  // BASE_PLANT_H_
