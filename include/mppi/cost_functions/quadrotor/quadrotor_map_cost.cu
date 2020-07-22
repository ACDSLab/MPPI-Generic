template <class CLASS_T, class PARAMS_T>
QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::QuadrotorMapCostImpl(cudaStream_t stream) {
  std::cout << "Hi there" << std::endl;
}

template <class CLASS_T, class PARAMS_T>
float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    const Eigen::Ref<const state_array> s, int timestep, int* crash_status) {
  std::cout << "It is a cost function" << std::endl;
  return 0;
}

template <class CLASS_T, class PARAMS_T>
__device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    float* s, int timestep, int* crash_status) {
  return 0;
}

template <class CLASS_T, class PARAMS_T>
float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::terminalCost(
    const Eigen::Ref<const state_array> s) {
  std::cout << "It is a cost function" << std::endl;
  return 0;
}

template <class CLASS_T, class PARAMS_T>
__device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::terminalCost(float* s) {
  return 0;
}