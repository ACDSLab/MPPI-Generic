//
// Created by jason on 12/12/22.
//

#include "racer_double_integrator_kinematic.cuh"

RacerDoubleIntegratorKinematic::RacerDoubleIntegratorKinematic(cudaStream_t stream)
  : MPPI_internal::Dynamics<RacerDoubleIntegratorKinematic, RacerDoubleIntegratorKinematicParams>(stream)
{
  this->requires_buffer_ = true;
  tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  steer_lstm_lstm_helper_ = std::make_shared<STEER_NN>(stream);
  delay_lstm_lstm_helper_ = std::make_shared<DELAY_NN>(stream);
  terra_lstm_lstm_helper_ = std::make_shared<TERRA_NN>(stream);
}

RacerDoubleIntegratorKinematic::state_array
RacerDoubleIntegratorKinematic::stateFromMap(const std::map<std::string, float>& map)
{
  return RacerDoubleIntegratorKinematic::state_array();
}
