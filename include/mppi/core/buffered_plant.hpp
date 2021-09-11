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
  typedef Eigen::Matrix<float, CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM, BUFFER_LENGTH+1> buffer_trajectory; // A control trajectory

  BufferedPlant(std::shared_ptr<CONTROLLER_T> controller, int hz, int opt_stide) :
  BasePlant<CONTROLLER_T>(controller, hz, opt_stide)
  {

  }

  void updateState(StateArray& state, double time) {
    // interpolates the control based off of the state timestamp
    BasePlant<CONTROLLER_T>::updateState(state, time);
    // add new state to the buffer

    // if the new time is older than the latest do not insert it
    if (time < prev_states_.back().second) {
      return;
    }
    prev_states_.push_back(std::make_pair<>(state, time));

    // remove old states from the buffer
    while(prev_states_.front().second < time - buffer_time_horizon_) {
      prev_states_.pop_front();
    }
  }

  buffer_trajectory getSmoothedBuffer() {
    // TODO create smooth sampling of the states using the latest state time

    std::array<std::vector<double>, CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM> states;
    std::vector<double> times;
    for(auto it = prev_states_.begin(); it != prev_states_.end(); it++) {
      StateArray s = it->first;
      double time = it->second;

      times.push_back(time);
      for(int i = 0; i < CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM; i++) {
        states[i].push_back(s(i));
      }
    }


    double final_time = times.back();

    // if the buffer isn't long enough return garbage
    if(final_time < *times.begin() + buffer_tau_) {
      return buffer_trajectory::Zero();
    }

    double start_time = final_time - buffer_tau_;
    std::vector<double> new_times;
    for(int i = 0; i < BUFFER_LENGTH+1; i++) {
      new_times.push_back(start_time + i*buffer_dt_);
    }

    buffer_trajectory result;

    std::vector<double> correct_knots;
    double knot_dt = (buffer_tau_)/(std::floor(times.size()/3) - 1);
    for(int i = 0; i < times.size()/3; i++) {
      correct_knots.push_back(start_time + i*knot_dt);
    }
    correct_knots.pop_back();
    correct_knots.erase(correct_knots.begin());

    for(int j = 0; j < CONTROLLER_T::TEMPLATED_DYNAMICS::STATE_DIM; j++) {
      auto curve = fitpackpp::BSplineCurve(times, states[j], correct_knots);
      std::vector<double> knots = curve.knotX();
      std::vector<double> coeff = curve.coefs();
      for(int i = 0; i < result.cols(); i++) {
        result(j, i) = curve.eval(new_times[i]);
      }
    }
    return result;
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
  std::list<std::pair<StateArray, double>> prev_states_;
  double buffer_time_horizon_ = 2.0; // how long to store values in the buffer
  double buffer_tau_ = 0.5; // how in history to create well sampled positions from
  double buffer_dt_ = 0.2; // the spacing between well sampled buffer positions
};


#endif //MPPIGENERIC_BUFFERED_PLANT_H
