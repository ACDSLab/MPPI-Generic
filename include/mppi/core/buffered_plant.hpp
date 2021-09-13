//
// Created by jason on 7/20/21.
//

#ifndef MPPIGENERIC_BUFFERED_PLANT_H
#define MPPIGENERIC_BUFFERED_PLANT_H

#include "base_plant.hpp"
#include <fitpackpp/bspline_curve.h>
#include <array>

template <class CONTROLLER_T, int BUFFER_LENGTH>
class BufferedPlant : public BasePlant<CONTROLLER_T> {
public:
  using StateArray = typename BasePlant<CONTROLLER_T>::s_array;
  using ControlArray = typename BasePlant<CONTROLLER_T>::c_array;
  static const int BUFFER_SIZE = CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM+CONTROLLER_T::TEMPLATED_DYNAMICS::CONTROL_DIM;
  typedef Eigen::Matrix<float, CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM+CONTROLLER_T::TEMPLATED_DYNAMICS::CONTROL_DIM, 1> buffer_state; //
  typedef Eigen::Matrix<float, CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM+CONTROLLER_T::TEMPLATED_DYNAMICS::CONTROL_DIM, BUFFER_LENGTH+1> buffer_trajectory; //

  BufferedPlant(std::shared_ptr<CONTROLLER_T> controller, int hz, int opt_stide) :
  BasePlant<CONTROLLER_T>(controller, hz, opt_stide)
  {

  }

  void updateState(StateArray& state, double time) {
    new_control_or_state_ = true;
    // interpolates the control based off of the state timestamp
    // TODO below are duplicated computations
    BasePlant<CONTROLLER_T>::updateState(state, time);
    double temp = this->last_used_pose_update_time_;
    double time_since_last_opt = time - this->last_used_pose_update_time_;

    //double temp = this->last_used_pose_update_time_ + this->controller_->getDt()*this->controller_->getNumTimesteps();
    bool t_within_trajectory = time >= this->last_used_pose_update_time_ &&
                               time < this->last_used_pose_update_time_ + this->controller_->getDt()*this->controller_->getNumTimesteps();
    if (!t_within_trajectory) {
      return;
    }
    StateArray target_nominal_state = this->controller_->interpolateState(this->state_traj_, time_since_last_opt);
    auto c_array = this->controller_->getCurrentControl(state, time_since_last_opt, target_nominal_state, this->control_traj_, this->feedback_gains_);

    // add new state to the buffer

    // if the new time is older than the latest do not insert it
    if (!prev_states_.empty() && time < prev_states_.back().second) {
      return;
    }
    buffer_state bufferState;
    for(int index = 0; index < CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM; index++) {
      bufferState(index, 0) = state(index);
    }
    for(int index = 0; index < CONTROLLER_T::TEMPLATED_DYNAMICS::CONTROL_DIM; index++) {
      bufferState(index+CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM, 0) = c_array(index);
    }
    prev_states_.push_back(std::make_pair<>(bufferState, time));

    // remove old states from the buffer
    int counter = 0;
    //printf("can pop %f < %f - %f => %f\n", prev_states_.front().second, time, buffer_time_horizon_, time - buffer_time_horizon_);
    while(prev_states_.front().second < time - buffer_time_horizon_) {
      prev_states_.pop_front();
      counter++;
    }

    // 2*k + 2 from curfit routine
    //printf("\n\nsize %d popped %d\n", prev_states_.size(), counter);
    if (prev_states_.size() > 8) {
      //printf("calling get smoothed buffer size %d", prev_states_.size());
      //std::cout << std::endl;
      getSmoothedBuffer();
    }
  }

  buffer_trajectory getSmoothedBuffer() {
    // TODO create smooth sampling of the states using the latest state time
    // TODO this cannot be multithreaded
    if(!new_control_or_state_) {
      std::cout << "not new control or state, skipping" << std::endl;
      return smoothed_buffer_;
    }

    std::array<std::vector<double>, BUFFER_SIZE> states;
    std::vector<double> times;
    for(auto it = prev_states_.begin(); it != prev_states_.end(); it++) {
      buffer_state s = it->first;
      double time = it->second;

      times.push_back(time);
      for(int i = 0; i < s.rows(); i++) {
        //printf("at state index %d pushing %f into buffer\n", i, s(i));
        states[i].push_back(s(i));
      }
    }

    double final_time = times.back();

    // if the buffer isn't long enough return garbage
    if(final_time < *times.begin() + buffer_tau_) {
      std::cout << "does not have a full buffer, exiting" << std::endl;
      return buffer_trajectory::Zero();
    }

    double start_time = final_time - buffer_tau_;
    std::vector<double> new_times;
    for(int i = 0; i < BUFFER_LENGTH+1; i++) {
      new_times.push_back(start_time + i*buffer_dt_);
    }

    std::vector<double> correct_knots;
    double knot_dt = (buffer_tau_)/(std::floor(times.size()/3) - 1);
    for(int i = 0; i < times.size()/3; i++) {
      std::cout << "knots " << i << " " << start_time + i*knot_dt << std::endl;
      correct_knots.push_back(start_time + i*knot_dt);
    }
    correct_knots.pop_back();
    correct_knots.erase(correct_knots.begin());

    for(int j = 0; j < smoothed_buffer_.rows(); j++) {
      //printf("at index %d: times size %d, states size %d, knots size %d", j, times.size(), states[j].size(), correct_knots.size());
      //std::cout << std::endl;
      auto curve = fitpackpp::BSplineCurve(times, states[j], correct_knots);
      std::vector<double> knots = curve.knotX();
      std::vector<double> coeff = curve.coefs();
      for(int i = 0; i < smoothed_buffer_.cols(); i++) {
        smoothed_buffer_(j, i) = curve.eval(new_times[i]);
      }
    }
    //std::cout << "smoothed buffer " << smoothed_buffer_ << std::endl;
    new_control_or_state_ = false;
    // TODO set new dynamics params and set flag to copy a initial hidden and cell state
    this->has_new_dynamics_params_ = true;
    this->dynamics_params_.updateBuffer(smoothed_buffer_);

    return smoothed_buffer_;
  }

  void setDynamicsParams(typename BasePlant<CONTROLLER_T>::DYN_PARAMS_T params) override {
    params.copy_everything = true;
    BasePlant<CONTROLLER_T>::setDynamicsParams(params);
  }

  double getEarliestTimeInBuffer() {
    if (prev_states_.empty()) {
      return -1;
    }
    return prev_states_.front().second;
  }

  double getLatestTimeInBuffer() {
    if (prev_states_.empty()) {
      return -1;
    }
    return prev_states_.back().second;
  }

  int getBufferSize() {
    return prev_states_.size();
  }

  void clearBuffer() {
    prev_states_.clear();
  }

protected:
  std::list<std::pair<buffer_state, double>> prev_states_;
  double buffer_time_horizon_ = 0.22; // how long to store values in the buffer
  double buffer_tau_ = 0.2; // how in history to create well sampled positions from
  double buffer_dt_ = 0.02; // the spacing between well sampled buffer positions

  std::atomic<bool> new_control_or_state_{false};
  buffer_trajectory smoothed_buffer_;
};


#endif //MPPIGENERIC_BUFFERED_PLANT_H
