#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>

template<class CLASS_T, class DYN_T, class PARAMS_T>
QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::QuadraticCostImpl(cudaStream_t stream) {
  this->bindToStream(stream);
}

template<class CLASS_T, class DYN_T, class PARAMS_T>
float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(
    const Eigen::Ref<const state_array> s, int timestep, int* crash_status) {
  float cost = 0;
  state_array des_state = this->params_.getDesiredState(timestep);

  Eigen::Matrix<float, DYN_T::STATE_DIM, DYN_T::STATE_DIM> coeffs;
  coeffs = Eigen::Matrix<float, DYN_T::STATE_DIM, DYN_T::STATE_DIM>::Zero();
  for (int i = 0; i < DYN_T::STATE_DIM; i++) {
    coeffs(i, i) = this->params_.s_coeffs[i];
  }
  state_array error = s - des_state;
  cost = error.transpose() * coeffs * error;

  return cost;
}

template<class CLASS_T, class DYN_T, class PARAMS_T>
float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(
    const Eigen::Ref<const state_array> s) {
  return 0.0;
}

template<class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(
    float* s, int timestep, int* crash_status) {
  float cost = 0;

  float * desired_state = this->params_.getGoalStatePointer(timestep);

  for(int i = 0; i < DYN_T::STATE_DIM; i++) {
    cost += powf(s[i] - desired_state[i], 2) * this->params_.s_coeffs[i];
  }

  return cost;
}

template<class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(
    float* s) {
  return 0.0;
}