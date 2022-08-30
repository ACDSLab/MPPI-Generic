//
// Created by jason on 7/20/21.
//

#ifndef MPPIGENERIC_BUFFERED_PLANT_H
#define MPPIGENERIC_BUFFERED_PLANT_H

#include "base_plant.hpp"
#include <array>
#include <algorithm>
#include <functional>

template <class T>
struct BufferMessage
{
  double time = -1;
  T data;
  bool required = true;

  BufferMessage(double time, T data)
  {
    this->time = time;
    this->data = data;
  }
};

template <class CONTROLLER_T>
class BufferedPlant : public BasePlant<CONTROLLER_T>
{
public:
  using s_array = typename BasePlant<CONTROLLER_T>::s_array;
  using c_array = typename BasePlant<CONTROLLER_T>::c_array;

  using buffer_trajectory = typename BasePlant<CONTROLLER_T>::buffer_trajectory;

  BufferedPlant(std::shared_ptr<CONTROLLER_T> controller, int hz, int opt_stide)
    : BasePlant<CONTROLLER_T>(controller, hz, opt_stide)
  {
  }

  void updateExtraValue(std::string name, float value, double time)
  {
    if (prev_extra_.find(name) == prev_extra_.end())
    {
      prev_extra_.emplace(std::make_pair(name, std::list<BufferMessage<float>>()));
    }
    insertionSort(prev_extra_[name], time, value);
  }

  void updateControls(c_array& control, double time)
  {
    insertionSort<c_array>(prev_controls_, time, control);
  }

  template <class T>
  void insertionSort(std::list<BufferMessage<T>>& list, double time, T val)
  {
    if (list.empty())
    {
      list.push_back(BufferMessage<T>(time, val));
      return;
    }
    for (auto it = list.rbegin(); it != list.rend(); it++)
    {
      if (it->time < time)
      {
        list.insert(it.base(), BufferMessage<T>(time, val));
        return;
      }
    }
    list.push_front(BufferMessage<T>(time, val));
  }

  template <class T>
  void cleanList(std::list<BufferMessage<T>>& list, double time)
  {
    if (list.empty())
    {
      return;
    }
    auto it = list.begin();
    // iterate until the time is greater than
    for (; (it != list.end() && it->time < time - buffer_time_horizon_); it++)
    {
    }

    list.erase(list.begin(), it);
  }

  static Eigen::Quaternionf interp(std::list<BufferMessage<Eigen::Quaternionf>>& list, double time)
  {
    if (list.empty())
    {
      return Eigen::Quaternionf::Identity();
    }
    auto it = list.rbegin();
    // iterate until the time is greater than
    for (; (it != list.rend() && it->time > time); it++)
    {
    }
    // if we search entire list then use the first index
    auto it_new = std::prev(it);
    if (it == list.rend())
    {
      return list.begin()->data;
    }

    double diff = it_new->time - it->time;

    double alpha = (time - it->time) / diff;
    // ensure we don't check beyond, i.e. use closest when interpolating outside bounds
    alpha = std::min(std::max(alpha, 0.0), 1.0);

    return it->data.slerp(alpha, it_new->data);
  }

  template <class T>
  static T interp(std::list<BufferMessage<T>>& list, double time)
  {
    if (list.empty())
    {
      return T();
    }
    auto it = list.rbegin();
    // iterate until the time is greater than
    for (; (it != list.rend() && it->time > time); it++)
    {
    }
    // if we search entire list then use the first index
    auto it_new = std::prev(it);
    if (it == list.rend())
    {
      return list.begin()->data;
    }

    double diff = it_new->time - it->time;

    double alpha = (time - it->time) / diff;
    // ensure we don't check beyond, i.e. use closest when interpolating outside bounds
    alpha = std::min(std::max(alpha, 0.0), 1.0);

    return (1 - alpha) * it->data + alpha * it_new->data;
  }

  void updateOdometry(Eigen::Vector3f pos, Eigen::Quaternionf quat, Eigen::Vector3f vel, Eigen::Vector3f omega,
                      double time)
  {
    // inserts odometry into buffers using insertion sort
    insertionSort(prev_position_, time, pos);
    insertionSort(prev_quaternion_, time, quat);
    insertionSort(prev_velocity_, time, vel);
    insertionSort(prev_omega_, time, omega);

    // grab the latest position update and pass to update state
    // gets the latest time for all information

    // TODO this interpolates using most recent if there are issues
    std::map<std::string, float> smoothed = getInterpState(time);
    s_array state = this->controller_->model_->stateFromMap(smoothed);
    BasePlant<CONTROLLER_T>::updateState(state, time);
  }

  double getStateTime()
  {
    if (prev_position_.empty())
    {
      return -1;
    }
    double time = prev_position_.back().time;
    time = std::min(prev_controls_.back().time, time);
    for (const auto& it : prev_extra_)
    {
      if (it.second.back().required)
      {
        time = std::min(it.second.back().time, time);
      }
    }
    return time;
  }

  std::map<std::string, float> getInterpState(double time)
  {
    std::map<std::string, float> result;
    Eigen::Vector3f interp_pos = interp(prev_position_, time);
    result["POS_X"] = interp_pos.x();
    result["POS_Y"] = interp_pos.y();
    result["POS_Z"] = interp_pos.z();
    Eigen::Quaternionf interp_quat = interp(prev_quaternion_, time);
    result["Q_W"] = interp_quat.w();
    result["Q_X"] = interp_quat.x();
    result["Q_Y"] = interp_quat.y();
    result["Q_Z"] = interp_quat.z();
    Eigen::Vector3f interp_vel = interp(prev_velocity_, time);
    result["VEL_X"] = interp_vel.x();
    result["VEL_Y"] = interp_vel.y();
    result["VEL_Z"] = interp_vel.z();
    Eigen::Vector3f interp_omega = interp(prev_omega_, time);
    result["OMEGA_X"] = interp_omega.x();
    result["OMEGA_Y"] = interp_omega.y();
    result["OMEGA_Z"] = interp_omega.z();

    auto controls_interp = interp(prev_controls_, time);
    for (int i = 0; i < controls_interp.rows(); i++)
    {
      result["CONTROL_" + std::to_string(i)] = controls_interp(i);
    }

    for (auto& it : prev_extra_)
    {
      result[it.first] = interp(it.second, time);
    }
    return result;
  }

  bool updateParameters()
  {
    // removes extra values from the buffer
    BasePlant<CONTROLLER_T>::updateParameters();
    std::lock_guard<std::mutex> guard(this->access_guard_);
    cleanBuffers();
  }

  buffer_trajectory getSmoothedBuffer()
  {
    double latest_time = getStateTime();

    std::lock_guard<std::mutex> lck(this->access_guard_);

    int steps = buffer_tau_ / buffer_dt_ + 1;

    buffer_trajectory result;

    // does the latest state to make sure we have valid values
    std::map<std::string, float> start_vals = getInterpState(latest_time);

    for (const auto& start_val : start_vals)
    {
      result[start_val.first] = Eigen::VectorXf(steps);
      result[start_val.first](steps - 1) = start_val.second;
    }

    // goes from [t - tau, t)
    for (int t = 0; t <= steps - 1; t++)
    {
      // get query time
      double query_time = latest_time - (steps - 1) * buffer_dt_ + t * buffer_dt_;
      std::map<std::string, float> interp_vals = getInterpState(query_time);

      // interpolate values
      for (auto& interp_val : interp_vals)
      {
        result[interp_val.first](t) = interp_val.second;
      }
    }

    return result;
  }

  void cleanBuffers()
  {
    double time = getStateTime();
    cleanList(prev_position_, time);
    cleanList(prev_quaternion_, time);
    cleanList(prev_velocity_, time);
    cleanList(prev_omega_, time);
    cleanList(prev_controls_, time);
    for (auto& it : prev_extra_)
    {
      cleanList(it.second, time);
    }
  }

  void cleanBuffers(double time)
  {
    cleanList(prev_position_, time);
    cleanList(prev_quaternion_, time);
    cleanList(prev_velocity_, time);
    cleanList(prev_omega_, time);
    cleanList(prev_controls_, time);
    for (auto& it : prev_extra_)
    {
      cleanList(it.second, time);
    }
  }

  void clearBuffers()
  {
    prev_position_.clear();
    prev_quaternion_.clear();
    prev_velocity_.clear();
    prev_omega_.clear();
    prev_controls_.clear();
    prev_extra_.clear();
  }

protected:
  std::list<BufferMessage<Eigen::Vector3f>> prev_position_;
  std::list<BufferMessage<Eigen::Quaternionf>> prev_quaternion_;
  std::list<BufferMessage<Eigen::Vector3f>> prev_velocity_;
  std::list<BufferMessage<Eigen::Vector3f>> prev_omega_;
  std::list<BufferMessage<c_array>> prev_controls_;
  std::map<std::string, std::list<BufferMessage<float>>> prev_extra_;

  double buffer_time_horizon_ = 2.0;   // how long to store values in the buffer
  double buffer_tau_ = 0.2;            // how in history to create well sampled positions from
  double buffer_dt_ = 0.02;            // the spacing between well sampled buffer positions
};

template class BufferMessage<Eigen::Vector3f>;
template class BufferMessage<Eigen::Quaternionf>;
template class BufferMessage<float>;

#endif  // MPPIGENERIC_BUFFERED_PLANT_H
