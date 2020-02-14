//
// Created by jason on 10/30/19.
//

#ifndef MPPIGENERIC_CONTROLLER_CUH
#define MPPIGENERIC_CONTROLLER_CUH

#include <array>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
class Controller {
public:
  Controller() = default;
  ~Controller() = default;
  /**
   * typedefs for access to templated class from outside classes
   */
  typedef DYN_T TEMPLATED_DYNAMICS;
  typedef COST_T TEMPLATED_COSTS;

  /**
   * Aliases
   */
   // Control
  typedef std::array<float, DYN_T::CONTROL_DIM> control_array; // State at a time t
  typedef std::array<float, DYN_T::CONTROL_DIM*MAX_TIMESTEPS> control_trajectory; // A control trajectory
  typedef std::array<float, DYN_T::CONTROL_DIM*MAX_TIMESTEPS*NUM_ROLLOUTS> sampled_control_traj; // All control trajectories sampled
  typedef std::array<float, DYN_T::CONTROL_DIM * DYN_T::STATE_DIM> K_matrix;

  // State
  typedef std::array<float, DYN_T::STATE_DIM> state_array; // State at a time t
  typedef std::array<float, DYN_T::STATE_DIM*MAX_TIMESTEPS> state_trajectory; // A state trajectory
  typedef std::array<float, DYN_T::STATE_DIM*MAX_TIMESTEPS*NUM_ROLLOUTS> sampled_state_traj; // All state trajectories sampled
  // Cost
  typedef std::array<float, MAX_TIMESTEPS> cost_trajectory; // A cost trajectory
  typedef std::array<float, NUM_ROLLOUTS> sampled_cost_traj; // All costs sampled for all rollouts

  /**
   * Public data members
   */
  DYN_T* model_;
  COST_T* cost_;

  /**
   * Given a state, calculates the optimal control sequence using MPPI according
   * to the cost function used as part of the template
   * @param state - the current state from which we would like to calculate
   * a control sequence
   */
  virtual void computeControl(state_array state) = 0;

  /**
   * Calculate new feedback gains
   */
  virtual void computeFeedbackGains(state_array state) {};

  /**
   * returns the current control sequence
   */
  virtual control_trajectory getControlSeq() = 0;

  /**
   * Return the current minimal cost sequence
   */
  virtual cost_trajectory getCostSeq() {
    return cost_trajectory();
  };

  /**
   * Return all the sampled costs sequences
   */
  virtual sampled_cost_traj getSampledCostSeq() {
    return sampled_cost_traj();
  };

  /**
   * Return control feedback gains
   */
  K_matrix getFeedbackGains() {
      K_matrix empty_feedback_gain = {{0}};
      return empty_feedback_gain;
  };

  /**
   * return the entire sample of control sequences
   */
  virtual sampled_control_traj getSampledControlSeq() {
    return sampled_control_traj();
  };

  /**
   * Return all the sampled states sequences
   */
  virtual sampled_state_traj getSampledStateSeq() {
    return sampled_state_traj();
  };

  /**
   * Return the current optimal state sequence
   */
  virtual state_trajectory getStateSeq() = 0;

  /**
   * Slide the control sequence back
   */
  virtual void slideControlSeq(int steps) {};

  /**
   * Reset Controls
   */
  virtual void resetControls() {};
};

#endif //MPPIGENERIC_CONTROLLER_CUH
