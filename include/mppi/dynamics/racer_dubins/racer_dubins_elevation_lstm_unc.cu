#include "racer_dubins_elevation_lstm_unc.cuh"

RacerDubinsElevationLSTMUncertainty::RacerDubinsElevationLSTMUncertainty(cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMUncertainty, RacerDubinsElevationParams>(stream)
{
  this->requires_buffer_ = true;
  // steer_helper_ = std::make_shared<LSTMLSTMHelper<>>(stream);
}

RacerDubinsElevationLSTMUncertainty::RacerDubinsElevationLSTMUncertainty(RacerDubinsElevationParams& params,
                                                                         cudaStream_t stream)
  : RacerDubinsElevationImpl<RacerDubinsElevationLSTMUncertainty, RacerDubinsElevationParams>(params, stream)
{
  this->requires_buffer_ = true;
  // steer_helper_ = std::make_shared<LSTMLSTMHelper<>>(stream);
  // uncertainty_helper_ = std::make_shared<LSTMLSTMHelper<>>(stream);
  // mean_helper_ = std::make_shared<LSTMLSTMHelper<>>(stream);
}

RacerDubinsElevationLSTMUncertainty::RacerDubinsElevationLSTMUncertainty(std::string path, cudaStream_t stream)
  : RacerDubinsElevationLSTMUncertainty(stream)
{
  if (!fileExists(path))
  {
    std::cerr << "Could not load neural net model at path: " << path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(path);
  // loads the steering constants
  this->params_.max_steer_rate = param_dict.at("steering/parameters/max_rate_pos").data<float>()[0];
  this->params_.steering_constant = param_dict.at("steering/parameters/constant").data<float>()[0];
  this->params_.steer_accel_constant = param_dict.at("steering/parameters/accel_constant").data<float>()[0];
  this->params_.steer_accel_drag_constant = param_dict.at("steering/parameters/accel_drag_constant").data<float>()[0];

  // steer_helper_->loadParams("steering/model");
  // uncertainty_helper_->loadParams("terra/uncertainty_network/model");
  // mean_helper_->loadParams("terra/mean_network/model");
}
