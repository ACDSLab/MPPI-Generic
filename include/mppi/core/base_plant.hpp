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

// TODO Figure out template here
template <class CONTROLLER_T>
class BasePlant {
public:
  using c_array = typename CONTROLLER_T::control_array;
  using c_traj = typename CONTROLLER_T::control_trajectory;

  using s_array = typename CONTROLLER_T::state_array;
  using s_traj = typename CONTROLLER_T::state_trajectory;
  using K_mat = typename CONTROLLER_T::K_matrix;

  using DYN_T = typename CONTROLLER_T::TEMPLATED_DYNAMICS;
  using DYN_PARAMS_T = typename DYN_T::DYN_PARAMS_T;
  using COST_T = typename CONTROLLER_T::TEMPLATED_COSTS;
  using COST_PARAMS_T = typename COST_T::COST_PARAMS_T;
protected:

  bool use_feedback_gains_ = false;
  int hz_ = 10; // Frequency of control publisher
  bool debug_mode_ = false;

  DYN_PARAMS_T dynamics_params_;
  COST_PARAMS_T cost_params_;

  bool has_new_dynamics_params_ = false;
  bool has_new_cost_params_ = false;

  // Values needed
  s_array init_state_ = s_array::Zero();
  c_array init_u_ = c_array::Zero();

  // Values updated at every time step
  s_array state_ = s_array::Zero();
  c_array u_ = c_array::Zero();
  s_traj state_traj_;
  c_traj control_traj_;

  // values sometime updated
  // TODO init to zero?
  K_mat feedback_gain_;

  // from ROSHandle mppi_node
  int optimization_stride_ = 1;

  /**
   * From before while loop
   */
  double last_pose_update_ = 0;
  double optimize_loop_time_ = 0; // duration of optimization loop (seconds)
  double avg_optimize_loop_time_ms_ = 0; //Average time between pose estimates
  double avg_optimize_tick_time_ms_ = 0; //Avg. time it takes to get to the sleep at end of loop
  double avg_sleep_time_ms_ = 0; //Average time spent sleeping
  //Counter, timing, and stride variables.
  int num_iter_ = 0;
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
//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BasePlant(int hz, int optimization_stride) {
    this->hz_ = hz;
    this->optimization_stride_ = optimization_stride;
  };
  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a BasePlant reference
   */
  virtual ~BasePlant() = default;

  /**
   * Gives the last time the pose was updated
   * @return the time at which the pose was upated in seconds
   */
  virtual double getLastPoseTime() = 0;

  /**
   * Return the latest state received
   * @return the latest state
   */
  virtual s_array getState() {return state_;};

  virtual void setState(s_array state) {state_ = state;}
  virtual void setControl(c_array u) {u_ = u;}
  virtual void setDebugMode(bool mode) {debug_mode_ = mode;}

  int getOptimizationStride() {return optimization_stride_;};
  void setOptimizationStride(int new_val) {optimization_stride_ = new_val;}

  virtual void getNewObstacles(std::vector<int>& obs_description,
                               std::vector<float>& obs_data) {};

  virtual void getNewCostmap(std::vector<int>& costmap_description,
                             std::vector<float>& costmap_data) {};

  // TODO should I keep this or not?
  virtual void getNewModel(std::vector<int>& model_description,
                           std::vector<float>& model_data) {};

  int getHz() {return hz_;}

  /**
   * Receives timing info from control loop and can be overwritten
   * to ouput to another system
   * @param avg_duration      Average duration of a single iteration in ms
   * @param avg_tick_duration [description]
   * @param avg_sleep_time    [description]
   */
  virtual void setTimingInfo(double avg_duration_ms,
                             double avg_tick_duration,
                             double avg_sleep_time) = 0;

  // TODO should publish to a topic not a static window
  /**
   * sets a debug image to be published at hz rate
   * @param debug_img
   */
  virtual void setDebugImage(cv::Mat debug_img) = 0;

  virtual void setSolution(const s_traj& state_seq,
                           const c_traj& control_seq,
                           const K_mat& feedback_gains,
                           double timestamp,
                           double loop_speed) = 0;



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
  * @brief Checks the system status.
  * @return An integer specifying the status. 0 means the system is operating
  * nominally, 1 means something is wrong but no action needs to be taken,
  * 2 means that the vehicle should stop immediately.
  */
  virtual int checkStatus() = 0;

  /**
   * Linearly interpolate the controls
   * @param  t           time in s that we desire a control for
   * @param  dt          time between control steps
   * @param  control_seq control trajectory used for interpolation
   * @return             the control at time t, interpolated from
   *                     the control sequence
   */
  c_array interpolateControls(double t,
                              double dt,
                              const c_traj& control_seq) {

    int lower_idx = (int) (t / dt);
    int upper_idx = lower_idx + 1;
    double alpha = (t - lower_idx * dt) / dt;

    c_array interpolated_control;
    int control_dim = CONTROLLER_T::TEMPLATED_DYNAMICS::CONTROL_DIM;
    for (int i = 0; i < interpolated_control.size(); i++) {
      float prev_cmd = control_seq(i, lower_idx);
      float next_cmd = control_seq(i, upper_idx);
      interpolated_control(i) = (1 - alpha) * prev_cmd + alpha * next_cmd;
    }
    return interpolated_control;
  };

  /**
   * Linearly interpolate the controls
   * @param  t           time in s that we desire a control for
   * @param  dt          time between control steps
   * @param  control_seq control trajectory used for interpolation
   * @return             the control at time t, interpolated from
   *                     the control sequence
   */
  // c_array interpolateFeedbackControls(double t,
  //                                     double dt,
  //                                     const c_traj& control_seq,
  //                                     const s_traj& state_seq,
  //                                     const s_array& curr_state,
  //                                     const K_mat& K) {
  //   int lower_idx = (int) (t / dt);
  //   int upper_idx = lower_idx + 1;
  //   double alpha = (t - lower_idx * dt) / dt;

  //   c_array interpolated_control;
  //   s_array desired_state;

  //   int state_dim = CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM;

  //   Eigen::MatrixXf current_state(7,1);
  //   Eigen::MatrixXf desired_state(7,1);
  //   Eigen::MatrixXf deltaU;
  //   current_state << full_state_.x_pos, full_state_.y_pos, full_state_.yaw, full_state_.roll, full_state_.u_x, full_state_.u_y, full_state_.yaw_mder;
  //   for (int i = 0; i < 7; i++){
  //     desired_state(i) = (1 - alpha)*stateSequence_[7*lowerIdx + i] + alpha*stateSequence_[7*upperIdx + i];
  //   }

  //   deltaU = ((1-alpha)*feedback_gains_[lowerIdx] + alpha*feedback_gains_[upperIdx])*(current_state - desired_state);

  //   if (std::isnan( deltaU(0) ) || std::isnan( deltaU(1))){
  //     steering = steering_ff;
  //     throttle = throttle_ff;
  //   }
  //   else {
  //     steering_fb = deltaU(0);
  //     throttle_fb = deltaU(1);
  //     steering = fmin(0.99, fmax(-0.99, steering_ff + steering_fb));
  //     throttle = fmin(throttleMax_, fmax(-0.99, throttle_ff + throttle_fb));
  //   }
  // };

  /**
   *
   * @param controller
   * @param is_alive
   * @return the milisecond number that the loop iteration started at
   */
  std::chrono::steady_clock::time_point runControlIteration(CONTROLLER_T* controller, std::atomic<bool>* is_alive) {
    if(!is_alive->load()) {
      // break out if it should stop
      return std::chrono::steady_clock::now();
    }

    double temp_last_pose_time = getLastPoseTime();

    // debug mode propagates dynamics on its own
    // TODO should not be debug mode, should be some mode that props dynamics anyway
    if (!debug_mode_){
      // wait for a new pose to compute control sequence from
      while(last_pose_update_ == temp_last_pose_time && is_alive->load()){
        usleep(50);
        temp_last_pose_time = getLastPoseTime();
      }
    }
    // TODO set up somewhere else
    std::chrono::milliseconds ms{(int)(optimization_stride_*1000.0/hz_)};
    optimize_loop_time_ = optimization_stride_ / (1.0 * hz_);

    std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
    num_iter_++;

    if (debug_mode_ && controller->cost_->getDebugDisplayEnabled()) { //Display the debug window.
      cv::Mat debug_img = controller->cost_->getDebugDisplay(state_.data());
      setDebugImage(debug_img);
    }

    //Update the state estimate
    if (last_pose_update_ != temp_last_pose_time){
      optimize_loop_time_ = temp_last_pose_time - last_pose_update_;
      last_pose_update_ = temp_last_pose_time;
      state_ = getState(); //Get the new state.
    }
    //Update the cost parameters
    if(has_new_cost_params_) {
      COST_PARAMS_T cost_params = getNewCostParams();
      controller->cost_->setParams(cost_params);
    }
    if (has_new_dynamics_params_) {
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
      // TODO define generic
      getNewCostmap(costmapDescription_, costmapData_);
      controller->cost_->updateCostmap(costmapDescription_, costmapData_);
    }

    //Figure out how many controls have been published since we were last here and slide the
    //control sequence by that much.
    int stride = round(optimize_loop_time_ * hz_);
    // std::cout << "Stride: " << stride << "," << optimizeLoopTime << std::endl;

    // TODO wat?
    if (status_ != 0){
      stride = optimization_stride_;
    }
    if (stride >= 0 && stride < controller->num_timesteps_){
      controller->slideControlSequence(stride);
    } else {
      // TODO
    }
    //Compute a new control sequence
    // std::cout << "BasePlant State: " << state.transpose() << std::endl;
    controller->computeControl(state_); //Compute the control

    control_traj_ = controller->getControlSeq();
    state_traj_ = controller->getStateSeq();
    // TODO should just be zerod out, not actually pull from anywhere
    if(use_feedback_gains_) {
      controller->computeFeedbackGains(state_);
      feedback_gain_ = controller->getFeedbackGains();
    }

    // std::cout << "Cost: " << controller->getBaselineCost() << std::endl;
    // std::cout << "BasePlant u(t = " << last_pose_update << ")\n" << control_traj << std::endl;

    //Set the updated solution for execution
    setSolution(state_traj_,
                control_traj_,
                feedback_gain_,
                last_pose_update_,
                avg_optimize_loop_time_ms_);

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

    std::chrono::duration<double, std::milli> loop_duration =
            std::chrono::steady_clock::now() - loop_start;
    double optimize_tick_time_ms = loop_duration.count();
    double projected_sleep_time_ms = (ms - loop_duration).count();

    // Update the average loop time data
    double prev_iter_percent = (num_iter_ - 1.0) / num_iter_;

    avg_optimize_loop_time_ms_ = prev_iter_percent * avg_optimize_loop_time_ms_ +
                              1000.0 * optimize_loop_time_ / num_iter_;
    avg_optimize_tick_time_ms_ = prev_iter_percent * avg_optimize_tick_time_ms_ +
                              optimize_tick_time_ms / num_iter_;
    avg_sleep_time_ms_ = prev_iter_percent * avg_sleep_time_ms_ +
                       projected_sleep_time_ms / num_iter_;

    setTimingInfo(avg_optimize_loop_time_ms_, avg_optimize_tick_time_ms_, avg_sleep_time_ms_);
    return loop_start;
  }

  void runControlLoop(CONTROLLER_T* controller,
                      std::atomic<bool>* is_alive) {
    //Initial condition of the robot
    state_ = init_state_;

    //Initial control value
    u_ = init_u_;

    last_pose_update_ = getLastPoseTime();
    optimize_loop_time_ = optimization_stride_ / (1.0 * hz_);

    //Set the loop rate
    std::chrono::milliseconds ms{(int)(optimization_stride_*1000.0/hz_)};
    if (!debug_mode_){
      while(last_pose_update_ == getLastPoseTime() && is_alive->load()){ //Wait until we receive a pose estimate
        usleep(50);
      }
    }
    controller->resetControls();
    //controller->computeFeedbackGains(state_);
    //Start the control loop.
    while (is_alive->load()) {
      std::chrono::steady_clock::time_point loop_start = runControlIteration(controller, is_alive);

      //Sleep for any leftover time in the control loop
      std::chrono::duration<double, std::milli> loop_duration =
        std::chrono::steady_clock::now() - loop_start;

      double optimizeTickTime_ms = loop_duration.count();

      double time_int = 1.0/hz_ - 0.0025;
      while(is_alive->load() &&
            (loop_duration < ms ||
              ((getLastPoseTime() - last_pose_update_) < time_int && status_ == 0))) {
        usleep(50);
        loop_duration = std::chrono::steady_clock::now() - loop_start;
      }
      double sleepTime_ms = loop_duration.count() - optimizeTickTime_ms;
    }
  }
};

#endif //BASE_PLANT_H_
