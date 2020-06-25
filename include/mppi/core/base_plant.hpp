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
#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <memory>

template <class CONTROLLER_T>
class BasePlant {
public:
  using c_array = typename CONTROLLER_T::control_array;
  using c_traj = typename CONTROLLER_T::control_trajectory;

  using s_array = typename CONTROLLER_T::state_array;
  using s_traj = typename CONTROLLER_T::state_trajectory;
  using K_traj = typename CONTROLLER_T::feedback_gain_trajectory;

  using DYN_T = typename CONTROLLER_T::TEMPLATED_DYNAMICS;
  using DYN_PARAMS_T = typename DYN_T::DYN_PARAMS_T;
  using COST_T = typename CONTROLLER_T::TEMPLATED_COSTS;
  using COST_PARAMS_T = typename COST_T::COST_PARAMS_T;
protected:

  std::mutex access_guard_;

  int hz_ = 10; // Frequency of control publisher
  bool debug_mode_ = false;
  bool increment_state_ = false;

  DYN_PARAMS_T dynamics_params_;
  COST_PARAMS_T cost_params_;

  bool has_new_dynamics_params_ = false;
  bool has_new_cost_params_ = false;
  bool recieved_debug_img_ = false;

  std::string debug_window_name_ = "NO NAME SET";
  cv::Mat debug_img_;

  // Values needed
  s_array init_state_ = s_array::Zero();
  c_array init_u_ = c_array::Zero();

  // Values updated at every time step
  s_array state_ = s_array::Zero();
  c_array u_ = c_array::Zero();
  // solution
  s_traj state_traj_;
  c_traj control_traj_;

  // values sometime updated
  // TODO init to zero?
  K_traj feedback_gains_;

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
  double last_used_pose_update_time_ = 0.0; // time of the last pose update that was used for optimization
  // Wall Clock: always real time
  double last_optimization_time_ = 0; // time of the last optimization
  double optimize_loop_duration_ = 0; // duration of the entire controller run loop
  double optimization_duration_ = 0; // Most recent time it took to run MPPI iteration
  double feedback_duration_ = 0; // most recent time it took to run the feedback controller
  double sleep_duration_ = 0; // how long the most recent loop in runControlLoop slept
  double avg_loop_time_ms_ = 0; // Average time it takes to run the controller
  double avg_optimize_time_ms_ = 0; // Average time it takes to runControlLoop
  double avg_feedback_time_ms_ = 0; // Average time it takes to run the feedback controller
  double avg_sleep_time_ms_ = 0; // Average time the runControlLoop sleeps between calls

  int num_iter_ = 0; // number of calls to computeControl
  /**
   * represents the status of the vehicle
   * 0: running normally
   * 1: not activated or no state information
   */
  int status_ = 1;

  //Obstacle and map parameters
  // TODO fix naming
  std::vector<int> obstacleDescription_;
  std::vector<float> obstacleData_;
  std::vector<int> costmapDescription_;
  std::vector<float> costmapData_;
  std::vector<int> modelDescription_;
  std::vector<float> modelData_;
public:
  std::shared_ptr<CONTROLLER_T> controller_;

//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BasePlant(std::shared_ptr<CONTROLLER_T> controller, int hz, int optimization_stride) {
    controller_ = controller;
    hz_ = hz;
    optimization_stride_ = optimization_stride;
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
   * Receives timing info from control loop and can be overwritten
   * to ouput to another system
   * @param avg_loop_ms          Average duration of a single iteration in ms
   * @param avg_optimize_ms      Average time to call computeControl
   * @param avg_feedback_ms      Average time to call computeFeedbackGains
   */
  virtual void setTimingInfo(double avg_loop_ms,
                             double avg_optimize_ms,
                             double avg_feedback_ms) = 0;

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

  // ======== PURE VIRTUAL END ====

  s_traj getStateTraj() {
    return state_traj_;
  }
  c_traj getControlTraj() {
    return control_traj_;
  }
  K_traj getFeedbackGains() {
    return feedback_gains_;
  }

  /**
   * Return the latest state received
   * @return the latest state
   */
  virtual s_array getState() {return state_;};

  virtual void setState(s_array state) {state_ = state;}
  virtual void setControl(c_array u) {u_ = u;}
  virtual void setDebugMode(bool mode) {debug_mode_ = mode;}

  int getTargetOptimizationStride() {return optimization_stride_;};
  int getLastOptimizationStride() {return last_optimization_stride_;};
  void setTargetOptimizationStride(int new_val) {optimization_stride_ = new_val;}

  virtual void getNewObstacles(std::vector<int>& obs_description,
                               std::vector<float>& obs_data) {};

  virtual void getNewCostmap(std::vector<int>& costmap_description,
                             std::vector<float>& costmap_data) {};

  // TODO should I keep this or not?
  virtual void getNewModel(std::vector<int>& model_description,
                           std::vector<float>& model_data) {};

  int getHz() {return hz_;}


  // TODO should publish to a topic not a static window
  /**
   * sets a debug image to be published at hz rate
   * @param debug_img
   */
  virtual void setDebugImage(cv::Mat debug_img, std::string name) {
    recieved_debug_img_ = true;
    std::lock_guard<std::mutex> guard(access_guard_);
    debug_window_name_ = name;
    debug_img_ = debug_img;
  }

  virtual void displayDebugImage() {
    if(recieved_debug_img_) {
      cv::namedWindow(debug_window_name_, cv::WINDOW_AUTOSIZE);
      cv::imshow(debug_window_name_, debug_img_);
      cv::waitKey(1);
    }
  }

  virtual void setSolution(const s_traj& state_seq,
                           const c_traj& control_seq,
                           const K_traj& feedback_gains,
                           double timestamp) {
    std::lock_guard<std::mutex> guard(access_guard_);
    last_used_pose_update_time_ = timestamp;
    state_traj_ = state_seq;
    control_traj_ = control_seq;
    feedback_gains_ = feedback_gains;
  }
  /**
   * updates the state and publishes a new control
   * @param state the most recent state from state estimator
   * @param time the time of the most recent state from the state estimator
   */
  virtual void updateState(s_array& state, double time) {
    // calculate and update all timing variables
    double time_since_last_opt = time - last_used_pose_update_time_;

    state_ = state;

    // check if the requested time is in the calculated trajectory
    bool t_within_trajectory = time > last_used_pose_update_time_ &&
                               time < last_used_pose_update_time_ + controller_->getDt()*controller_->getNumTimesteps();

    if (time_since_last_opt > 0 && t_within_trajectory){
      pubControl(controller_->getCurrentControl(state, time_since_last_opt));
    }
  }

  virtual bool hasNewDynamicsParams() {return has_new_dynamics_params_;};
  virtual bool hasNewCostParams() {return has_new_cost_params_;};

  virtual DYN_PARAMS_T getNewDynamicsParams() {
    has_new_dynamics_params_ = false;
    return dynamics_params_;
  }
  virtual COST_PARAMS_T getNewCostParams() {
    has_new_cost_params_ = false;
    return cost_params_;
  }

  virtual void setDynamicsParams(DYN_PARAMS_T params) {
    dynamics_params_ = params;
    has_new_dynamics_params_ = true;
  }
  virtual void setCostParams(COST_PARAMS_T params) {
    cost_params_ = params;
    has_new_cost_params_ = true;
  }

  virtual bool hasNewObstacles() { return false;};
  virtual bool hasNewCostmap() { return false;};
  virtual bool hasNewModel() { return false;};

  /**
   *
   * @param controller
   * @param state
   * @return
   */
  bool updateParameters(CONTROLLER_T* controller, s_array& state) {
    bool changed = false;
    if (debug_mode_ && controller->cost_->getDebugDisplayEnabled()) { //Display the debug window.
      changed = true;
      cv::Mat debug_img = controller->cost_->getDebugDisplay(state.data());
      setDebugImage(debug_img, debug_window_name_);
    }
    //Update the cost parameters
    if(hasNewCostParams()) {
      changed = true;
      COST_PARAMS_T cost_params = getNewCostParams();
      controller->cost_->setParams(cost_params);
    }
    // update dynamics params
    if (hasNewDynamicsParams()) {
      changed = true;
      DYN_PARAMS_T dyn_params = getNewDynamicsParams();
      controller->model_->setParams(dyn_params);
    }
    //Update any obstacles
    /*
    TODO should this exist at all?
    if (hasNewObstacles()){
      getNewObstacles(obstacleDescription, obstacleData);
      controller->cost_->updateObstacles(obstacleDescription, obstacleData);
    }
     */
    //Update the costmap
    if (hasNewCostmap()){
      changed = true;
      // TODO define generic
      getNewCostmap(costmapDescription_, costmapData_);
      controller->cost_->updateCostmap(costmapDescription_, costmapData_);
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
  void runControlIteration(CONTROLLER_T* controller, std::atomic<bool>* is_alive) {
    if(!is_alive->load()) {
      // break out if it should stop
      return;
    }

    double temp_last_pose_time = getCurrentTime();

    // debug mode propagates dynamics on its own
    if (!debug_mode_){
      // wait for a new pose to compute control sequence from
      while(last_used_pose_update_time_ == temp_last_pose_time && is_alive->load()) {
        usleep(50);
        temp_last_pose_time = getCurrentTime();
      }
    }

    std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();

    s_array state = getState();
    num_iter_++;

    // calculate how much we should slide the control sequence
    double dt = last_used_pose_update_time_ - temp_last_pose_time;
    if(last_used_pose_update_time_ == 0) {
      // should only happen on the first iteration
      dt = 0;
      last_optimization_stride_ = 0;
    } else {
      last_optimization_stride_ = std::max(int(round(dt / hz_)), optimization_stride_);
    }
    last_used_pose_update_time_ = temp_last_pose_time;
    // determine how long we should stride based off of robot time

    // TODO call update importance sampler

    if (last_optimization_stride_ > 0 && last_optimization_stride_ < controller->num_timesteps_){
      controller->slideControlSequence(last_optimization_stride_);
    }

    // Compute a new control sequence
    std::chrono::steady_clock::time_point optimization_start = std::chrono::steady_clock::now();
    controller->computeControl(state); // Compute the nominal control sequence

    c_traj control_traj = controller->getControlSeq();
    s_traj state_traj = controller->getStateSeq();
    optimization_duration_ = (std::chrono::steady_clock::now() - optimization_start).count() / 1e6;

    std::chrono::steady_clock::time_point feedback_start = std::chrono::steady_clock::now();
    // TODO make sure this is zero by default
    K_traj feedback_gains;
    if(controller->getFeedbackEnabled()) {
      controller->computeFeedbackGains(state);
      feedback_gains = controller->getFeedbackGains();
    }
    feedback_duration_ = (std::chrono::steady_clock::now() - feedback_start).count() / 1e6;

    //Set the updated solution for execution
    setSolution(state_traj,
                control_traj,
                feedback_gains,
                temp_last_pose_time);

    //Check the robots status
    status_ = checkStatus();

    // TODO
    //Increment the state if debug mode is set to true
    // if (status != 0 && debug_mode_){
    //   for (int t = 0; t < optimization_stride; t++){
    //     int control_dim = CONTROLLER_T::TEMPLATED_DYNAMICS::CONTROL_DIM;
    //     for (int i = 0; i < control_dim; i++) {
    //       u[i] = control_traj[control_dim * t + i];
    //     }
    //     controller->model_->updateState(state, u);
    //   }
    // }

    optimize_loop_duration_ = (std::chrono::steady_clock::now() - loop_start).count() / 1e6;

    // Update the average loop time data
    double prev_iter_percent = (num_iter_ - 1.0) / num_iter_;

    avg_loop_time_ms_ = prev_iter_percent * avg_loop_time_ms_ +
                              optimize_loop_duration_ / num_iter_;
    avg_optimize_time_ms_ = prev_iter_percent * avg_optimize_time_ms_ +
                              optimization_duration_ / num_iter_;
    avg_feedback_time_ms_ = prev_iter_percent * avg_feedback_time_ms_ +
                              feedback_duration_ /num_iter_;

    setTimingInfo(avg_loop_time_ms_, avg_optimize_time_ms_, avg_feedback_time_ms_);
  }

  void runControlLoop(CONTROLLER_T* controller,
                      std::atomic<bool>* is_alive) {
    //Initial condition of the robot
    state_ = init_state_;

    //Initial control value
    u_ = init_u_;

    double temp_last_pose_time = getCurrentTime();

    //Set the loop rate
    std::chrono::milliseconds ms{(int)(optimization_stride_*1000.0/hz_)};
    if (!debug_mode_){
      while(last_used_pose_update_time_ == temp_last_pose_time && is_alive->load()){
        usleep(50);
        temp_last_pose_time = getCurrentTime();
      }
    }
    controller->resetControls();
    last_used_pose_update_time_ = getCurrentTime();

    //Start the control loop.
    while (is_alive->load()) {
      runControlIteration(controller, is_alive);

      double wait_until_time = last_used_pose_update_time_ + (1.0/hz_)*last_optimization_stride_;

      std::chrono::steady_clock::time_point sleep_start = std::chrono::steady_clock::now();
      while(is_alive->load() && status_ == 0 && wait_until_time > getCurrentTime()) {
        usleep(50);
      }
      sleep_duration_ = (std::chrono::steady_clock::now() - sleep_start).count() / 1e6;
      double prev_iter_percent = (num_iter_ - 1.0) / num_iter_;
      avg_sleep_time_ms_ = prev_iter_percent * avg_sleep_time_ms_ +
              sleep_duration_/num_iter_;
    }
  }
};

#endif //BASE_PLANT_H_