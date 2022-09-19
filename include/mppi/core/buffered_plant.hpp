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

template <class DYN_T>
class Buffer
{
public:
  using buffer_trajectory = typename DYN_T::buffer_trajectory;
  using c_array = typename DYN_T::control_array;

  void updateExtraValue(const std::string& name, float value, double time)
  {
    std::lock_guard<std::mutex> guard(this->buffer_guard_);
    if (prev_extra_.find(name) == prev_extra_.end())
    {
      prev_extra_.emplace(std::make_pair(name, std::list<BufferMessage<float>>()));
    }
    insertionSort(prev_extra_[name], time, value);
  }

  void updateControls(c_array& control, double time)
  {
    std::lock_guard<std::mutex> guard(this->buffer_guard_);
    insertionSort<c_array>(prev_controls_, time, control);
  }

  template <class T>
  static void insertionSort(std::list<BufferMessage<T>>& list, double time, T val)
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
  static void cleanList(std::list<BufferMessage<T>>& list, double time, double buffer_time_horizon)
  {
    if (list.empty())
    {
      return;
    }
    auto it = list.begin();
    // iterate until the time is greater than
    for (; (it != list.end() && it->time < time - buffer_time_horizon); it++)
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
    if (it == list.rbegin())
    {
      return it->data;
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
    if (it == list.rbegin())
    {
      return it->data;
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

  void updateOdometry(Eigen::Vector3f& pos, Eigen::Quaternionf& quat, Eigen::Vector3f& vel, Eigen::Vector3f& omega,
                      double time)
  {
    this->buffer_guard_.lock();
    // inserts odometry into buffers using insertion sort
    insertionSort(prev_position_, time, pos);
    insertionSort(prev_quaternion_, time, quat);
    insertionSort(prev_velocity_, time, vel);
    insertionSort(prev_omega_, time, omega);
    this->buffer_guard_.unlock();
  }

  std::map<std::string, float> getInterpState(double time)
  {
    std::lock_guard<std::mutex> guard(this->buffer_guard_);
    std::map<std::string, float> result;
    if (prev_position_.empty())
    {
      return result;
    }

    Eigen::Vector3f interp_pos = interp(prev_position_, time);
    result["POS_X"] = interp_pos.x();
    result["POS_Y"] = interp_pos.y();
    result["POS_Z"] = interp_pos.z();
    Eigen::Quaternionf interp_quat = interp(prev_quaternion_, time);
    result["Q_W"] = interp_quat.w();
    result["Q_X"] = interp_quat.x();
    result["Q_Y"] = interp_quat.y();
    result["Q_Z"] = interp_quat.z();
    float roll, pitch, yaw;
    mppi::math::Quat2EulerNWU(interp_quat, roll, pitch, yaw);
    result["ROLL"] = roll;
    result["PITCH"] = pitch;
    result["YAW"] = yaw;
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

  buffer_trajectory getSmoothedBuffer(double latest_time, double buffer_tau, double buffer_dt)
  {
    // does the latest state to make sure we have valid values
    std::map<std::string, float> start_vals = getInterpState(latest_time);
    buffer_trajectory result;

    // if not enough data return empty message
    this->buffer_guard_.lock();
    if (prev_position_.rbegin()->time - prev_position_.begin()->time < buffer_tau)
    {
      std::cout << "not enough time for buffer, returning early" << prev_position_.rbegin()->time << " - "
                << prev_position_.begin()->time << " < " << buffer_tau << std::endl;
      this->buffer_guard_.unlock();
      return result;
    }
    this->buffer_guard_.unlock();

    int steps = buffer_tau / buffer_dt + 1;

    for (const auto& start_val : start_vals)
    {
      result[start_val.first] = Eigen::VectorXf(steps);
      result[start_val.first](steps - 1) = start_val.second;
    }

    // goes from [t - tau, t)
    for (int t = 0; t <= steps - 1; t++)
    {
      // get query time
      double query_time = latest_time - (steps - 1) * buffer_dt + t * buffer_dt;
      std::map<std::string, float> interp_vals = getInterpState(query_time);

      // interpolate values
      for (auto& interp_val : interp_vals)
      {
        result[interp_val.first](t) = interp_val.second;
      }
    }

    return result;
  }

  void cleanBuffers(double time, double horizon)
  {
    std::lock_guard<std::mutex> guard(this->buffer_guard_);
    cleanList(prev_position_, time, horizon);
    cleanList(prev_quaternion_, time, horizon);
    cleanList(prev_velocity_, time, horizon);
    cleanList(prev_omega_, time, horizon);
    cleanList(prev_controls_, time, horizon);
    for (auto& it : prev_extra_)
    {
      cleanList(it.second, time, horizon);
    }
  }

  void clearBuffers()
  {
    std::lock_guard<std::mutex> guard(this->buffer_guard_);
    prev_position_.clear();
    prev_quaternion_.clear();
    prev_velocity_.clear();
    prev_omega_.clear();
    prev_controls_.clear();
    prev_extra_.clear();
  }

  std::list<BufferMessage<Eigen::Vector3f>> getPrevPositionList()
  {
    return prev_position_;
  }

  std::list<BufferMessage<Eigen::Quaternionf>> getPrevQuaternionList()
  {
    return prev_quaternion_;
  }

  std::list<BufferMessage<Eigen::Vector3f>> getPrevVelocityList()
  {
    return prev_velocity_;
  }

  std::list<BufferMessage<Eigen::Vector3f>> getPrevOmegaList()
  {
    return prev_omega_;
  }

  std::list<BufferMessage<c_array>> getPrevControlList()
  {
    return prev_controls_;
  }

  std::map<std::string, std::list<BufferMessage<float>>> getPrevExtraList()
  {
    return prev_extra_;
  }

private:
  std::mutex buffer_guard_;

  std::list<BufferMessage<Eigen::Vector3f>> prev_position_;
  std::list<BufferMessage<Eigen::Quaternionf>> prev_quaternion_;
  std::list<BufferMessage<Eigen::Vector3f>> prev_velocity_;
  std::list<BufferMessage<Eigen::Vector3f>> prev_omega_;
  std::list<BufferMessage<c_array>> prev_controls_;
  std::map<std::string, std::list<BufferMessage<float>>> prev_extra_;
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

  void updateExtraValue(const std::string& name, float value, double time)
  {
    buffer_.updateExtraValue(name, value, time);
  }

  void updateControls(c_array& control, double time)
  {
    buffer_.updateControls(control, time);
  }

  void updateOdometry(Eigen::Vector3f& pos, Eigen::Quaternionf& quat, Eigen::Vector3f& vel, Eigen::Vector3f& omega,
                      double time)
  {
    buffer_.updateOdometry(pos, quat, vel, omega, time);

    /**
     * Uses the most recent odometry information
     * If any other sources are more delayed it uses the most recent value
     * If other sources are more recent it gets interpolated to odom time
     */
    std::map<std::string, float> smoothed = buffer_.getInterpState(time);
    s_array state = this->controller_->model_->stateFromMap(smoothed);
    BasePlant<CONTROLLER_T>::updateState(state, time);
  }

  std::map<std::string, float> getInterpState(double time)
  {
    return buffer_.getInterpState(time);
  }

  bool updateParameters()
  {
    // removes extra values from the buffer
    double time = this->getStateTime();
    buffer_.cleanBuffers(time, buffer_time_horizon_);
    return BasePlant<CONTROLLER_T>::updateParameters();
  }

  buffer_trajectory getSmoothedBuffer(double latest_time)
  {
    return buffer_.getSmoothedBuffer(latest_time, buffer_tau_, buffer_dt_);
  }

  void cleanBuffers(double time)
  {
    buffer_.cleanBuffers(time, buffer_time_horizon_);
  }

  void clearBuffers()
  {
    buffer_.clearBuffers();
  }

protected:
  Buffer<typename CONTROLLER_T::TEMPLATED_DYNAMICS> buffer_;

  double buffer_time_horizon_ = 2.0;   // how long to store values in the buffer
  double buffer_tau_ = 1.0;            // how in history to create well sampled positions from
  double buffer_dt_ = 0.02;            // the spacing between well sampled buffer positions
};

template class BufferMessage<Eigen::Vector3f>;
template class BufferMessage<Eigen::Quaternionf>;
template class BufferMessage<float>;

#endif  // MPPIGENERIC_BUFFERED_PLANT_H
