//
// Created by jason on 10/30/19.
//

#ifndef MPPIGENERIC_CONTROLLER_CUH
#define MPPIGENERIC_CONTROLLER_CUH

#include <array>
#include <Eigen/Core>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
class Controller {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Controller() = default;
  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a basePlant reference
   */
  virtual ~Controller() = default;
  /**
   * typedefs for access to templated class from outside classes
   */
  typedef DYN_T TEMPLATED_DYNAMICS;
  typedef COST_T TEMPLATED_COSTS;

  /**
   * Aliases
   */
   // Control
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, MAX_TIMESTEPS> control_trajectory; // A control trajectory
//  typedef util::NamedEigenAlignedVector<control_trajectory> sampled_control_traj;
//  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> K_matrix;

  // State
  using state_array = typename DYN_T::state_array;
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM, MAX_TIMESTEPS> state_trajectory; // A state trajectory
//  typedef util::NamedEigenAlignedVector<state_trajectory> sampled_state_traj;

  // Cost
  typedef Eigen::Matrix<float, MAX_TIMESTEPS, 1> cost_trajectory;
  typedef Eigen::Matrix<float, NUM_ROLLOUTS, 1> sampled_cost_traj;
//  typedef std::array<float, MAX_TIMESTEPS> cost_trajectory; // A cost trajectory
//  typedef std::array<float, NUM_ROLLOUTS> sampled_cost_traj; // All costs sampled for all rollouts

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
  virtual void computeControl(const Eigen::Ref<const state_array>& state) = 0;

  /**
   * Calculate new feedback gains
   */
  virtual void computeFeedbackGains(const Eigen::Ref<const state_array>& state) {};

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
//  K_matrix getFeedbackGains() {
//      K_matrix empty_feedback_gain;
//      return empty_feedback_gain;
//  };

  /**
   * return the entire sample of control sequences
   */
//  virtual sampled_control_traj getSampledControlSeq() {
//    return sampled_control_traj();
//  };

  /**
   * Return all the sampled states sequences
   */
//  virtual sampled_state_traj getSampledStateSeq() {
//    return sampled_state_traj();
//  };

  /**
   * Return the current optimal state sequence
   */
  virtual state_trajectory getStateSeq() = 0;

  /**
   * Slide the control sequence back
   */
  virtual void slideControlSequence(int steps) = 0;

  /**
   * Reset Controls
   */
  virtual void resetControls() {};
};

#endif //MPPIGENERIC_CONTROLLER_CUH
