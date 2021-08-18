//
// Created by jason on 7/20/21.
//

#ifndef MPPIGENERIC_BUFFERED_PLANT_H
#define MPPIGENERIC_BUFFERED_PLANT_H

#include "base_plant.hpp"

template <class CONTROLLER_T>
class BufferedPlant : public BasePlant<CONTROLLER_T> {
public:
  using StateArray = typename BasePlant<CONTROLLER_T>::s_array;

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

  StateArray getState() override {
    // TODO create smooth sampling of the states using the latest state time
  }

  double getEarliestTimeInBuffer() {
    if (prev_states_.size() == 0) {
      return -1;
    }
    return prev_states_.front().second;
  }

  double getLatestTimeInBuffer() {
    if (prev_states_.size() == 0) {
      return -1;
    }
    return prev_states_.back().second;
  }

  int getBufferSize() {
    return prev_states_.size();
  }

protected:
  std::list<std::pair<StateArray, double>> prev_states_;
  double buffer_time_horizon_ = 2.0; // how long to store values in the buffer
  double buffer_tau_ = 0.5; // how in history to create well sampled positions from
  double buffer_dt_ = 0.2; // the spacing between well sampled buffer positions
};


#endif //MPPIGENERIC_BUFFERED_PLANT_H
