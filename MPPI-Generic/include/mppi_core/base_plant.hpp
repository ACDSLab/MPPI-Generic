/**
 * Created by jason on 10/30/19.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
**/

#ifndef BASE_PLANT_H_
#define BASE_PLANT_H_

#include "mppi_common.cuh"

// Double check if these are included in mppi_common.h
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <chrono>
#include <atomic>

// TODO Figure out template here
template <class CONTROLLER_T>
class basePlant {
private:
  // from SystemParams * params
  int hz = 0;
  bool debug_mode = false;
  int num_timesteps = 0;

  // Values needed
  CONTROLLER_T::state_t init_state = {0};
  CONTROLLER_T::control_t init_u = {0};

  // Values updated at every time step
  CONTROLLER_T::state_t state;
  CONTROLLER_T::control_t u;
  CONTROLLER_T::state_trajectory state_traj;
  CONTROLLER_T::control_trajectory control_traj;

  // from ROSHandle mppi_node
  int optimization_stride = 0;
  bool use_feedback_gains = 0;

  /**
   * From before while loop
   */
  double last_pose_update = 0;
  double optimizeLoopTime = 0; // duration of optimization loop (seconds)
  double avgOptimizeLoopTime_ms = 0; //Average time between pose estimates
  double avgOptimizeTickTime_ms = 0; //Avg. time it takes to get to the sleep at end of loop
  double avgSleepTime_ms = 0; //Average time spent sleeping
  //Counter, timing, and stride variables.
  int num_iter = 0;
  int status = 1;

  //Obstacle and map parameters
  std::vector<int> obstacleDescription;
  std::vector<float> obstacleData;
  std::vector<int> costmapDescription;
  std::vector<float> costmapData;
  std::vector<int> modelDescription;
  std::vector<float> modelData;
public:

  /**
   * Gives the last time the pose was updated
   * @return the time at which the pose was upated in seconds
   */
  virtual double getLastPoseTime() = 0;

  /**
   * Return the latest state received
   * @return the latest state
   */
  virtual CONTROLLER_T::state_t getState();

  // TODO: Set Params typedef in Costs.cu after merge with Jason's branch
  // virtual typename CONTROLLER_T::TEMPLATED_COSTS::TEMPLATED_PARAMS getDynRcfgParams();

  virtual void getNewObstacles(std::vector<int>& obs_description,
                               std::vector<float>& obs_data);

  virtual void getNewCostmap(std::vector<int>& costmap_description,
                             std::vector<float>& costmap_data);

  virtual void getNewModel(std::vector<int>& model_description,
                           std::vector<float>& model_data);

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

  virtual void setDebugImage(cv::Mat debug_img) = 0;

  virtual void setSolution(CONTROLLER_T::state_trajectory state_seq,
                           CONTROLLER_T::control_trajectory control_seq,
                           CONTROLLER_T::K_matrix feedback_gains,
                           double timestamp,
                           double loop_speed) = 0;




  virtual bool hasNewDynRcfg() { return false;};
  virtual bool hasNewObstacles() { return false;};
  virtual bool hasNewCostmap() { return false;};
  virtual bool hasNewModel() { return false;};

  /**
  * @brief Checks the system status.
  * @return An integer specifying the status. 0 means the system is operating
  * nominally, 1 means something is wrong but no action needs to be taken,
  * 2 means that the vehicle should stop immediately.
  */
  virtual int checkStatus() {return 0;};

  void runControlLoop(CONTROLLER_T* controller,
                      std::atomic<bool>* is_alive) {
    //Initial condition of the robot
    state = init_state;

    //Initial control value
    u = init_u;

    last_pose_update = getLastPoseTime();
    optimizeLoopTime = optimization_stride / (1.0 * hz);

    //Set the loop rate
    std::chrono::milliseconds ms{(int)(optimization_stride*1000.0/params->hz)};

    if (!params->debug_mode){
      while(last_pose_update == getLastPoseTime() && is_alive->load()){ //Wait until we receive a pose estimate
        usleep(50);
      }
    }

    controller->resetControls();
    controller->computeFeedbackGains(state);

    //Start the control loop.
    while (is_alive->load()) {
      std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
      setTimingInfo(avgOptimizeLoopTime_ms, avgOptimizeTickTime_ms, avgSleepTime_ms);
      num_iter ++;

      if (debug_mode){ //Display the debug window.
        // TODO: have getDebugDisplay take in entire state, not just 3 values
        cv::Mat debug_img = controller->costs_->getDebugDisplay(state(0), state(1), state(2));
        setDebugImage(debug_img);
      }
      //Update the state estimate
      if (last_pose_update != getLastPoseTime()){
        optimizeLoopTime = getLastPoseTime() - last_pose_update;
        last_pose_update = getLastPoseTime();
        state = getState(); //Get the new state.
      }
      //Update the cost parameters
      if (hasNewDynRcfg()){
        // TODO: get parameters typedef from costs
        // controller->costs_->updateParams_dcfg(getDynRcfgParams());
      }
      //Update any obstacles
      if (hasNewObstacles()){
        getNewObstacles(obstacleDescription, obstacleData);
        controller->costs_->updateObstacles(obstacleDescription, obstacleData);
      }
      //Update the costmap
      if (hasNewCostmap()){
        getNewCostmap(costmapDescription, costmapData);
        controller->costs_->updateCostmap(costmapDescription, costmapData);
      }
      //Update dynamics model
      if (hasNewModel()){
        getNewModel(modelDescription, modelData);
        controller->model_->updateModel(modelDescription, modelData);
      }

      //Figure out how many controls have been published since we were last here and slide the
      //control sequence by that much.
      int stride = round(optimizeLoopTime * hz);
      if (status != 0){
        stride = optimization_stride;
      }
      if (stride >= 0 && stride < num_timesteps){
        controller->slideControlSeq(stride);
      }

      //Compute a new control sequence
      controller->computeControl(state); //Compute the control
      if (use_feedback_gains){
        controller->computeFeedbackGains(state);
      }
      control_traj = controller->getControlSeq();
      strate_traj = controller->getStateSeq();
      CONTROLLER_T::K_matrix feedback_gain = controller->getFeedbackGains();

      //Set the updated solution for execution
      setSolution(strate_traj,
                  control_traj,
                  feedback_gain,
                  last_pose_update,
                  avgOptimizeLoopTime_ms);

      //Check the robots status
      status = checkStatus();

      //Increment the state if debug mode is set to true
      if (status != 0 && debug_mode){
        for (int t = 0; t < optimization_stride; t++){
          u << controlSolution[2*t], controlSolution[2*t + 1];
          controller->model_->updateState(state, u);
        }
      }

      //Sleep for any leftover time in the control loop
      std::chrono::duration<double, std::milli> fp_ms =
        std::chrono::steady_clock::now() - loop_start;

      double optimizeTickTime_ms = fp_ms.count();
      int count = 0;

      double time_int = 1.0/hz - 0.0025;
      while(is_alive->load() &&
            (fp_ms < ms ||
              ((getLastPoseTime() - last_pose_update) < time_int && status == 0))) {
        usleep(50);
        fp_ms = std::chrono::steady_clock::now() - loop_start;
        count++;
      }
      double sleepTime_ms = fp_ms.count() - optimizeTickTime;

      // Update the average loop time data
      double prev_iter_percent = (num_iter - 1.0) / num_iter;

      avgOptimizeLoopTime_ms = prev_iter_percent * avgOptimizeLoopTime_ms +
       1000.0 * optimizeLoopTime / num_iter;

      avgOptimizeTickTime_ms = prev_iter_percent * avgOptimizeTickTime_ms +
        optimizeTickTime_ms / num_iter;

      avgSleepTime_ms = prev_iter_percent * avgSleepTime_ms +
        sleepTime_ms / num_iter;
    }
  }
};

#endif //BASE_PLANT_H_
