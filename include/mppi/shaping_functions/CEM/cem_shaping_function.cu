template<class CLASS_T, class PARAMS_T, int NUM_ROLLOUTS, int BDIM_X>
__host__ __device__ float CEMShapingFunctionImpl<CLASS_T, PARAMS_T, NUM_ROLLOUTS, BDIM_X>::computeWeight(float* traj_cost,
                                                                                                      float baseline, int global_idx) {
  return traj_cost[global_idx] > baseline ? (1.0/((int) NUM_ROLLOUTS*this->params_.gamma)) : 0;
}

template<class CLASS_T, class PARAMS_T, int NUM_ROLLOUTS, int BDIM_X>
void CEMShapingFunctionImpl<CLASS_T, PARAMS_T, NUM_ROLLOUTS, BDIM_X>::computeWeights(cost_traj& trajectory_costs,
                                                                                  float* trajectory_costs_d, cudaStream_t stream) {
  // Copy the costs back to the host
  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs.data(),
                               trajectory_costs_d,
                               NUM_ROLLOUTS*sizeof(float),
                               cudaMemcpyDeviceToHost, this->stream_));
  HANDLE_ERROR( cudaStreamSynchronize(stream) );

  std::nth_element(trajectory_costs.data(),
                   trajectory_costs.data() + (int) (NUM_ROLLOUTS*this->params_.gamma),
                   trajectory_costs.data()+NUM_ROLLOUTS, std::greater<float>());
  this->baseline_ = trajectory_costs((int) NUM_ROLLOUTS*this->params_.gamma);

          // launch the WeightKernel to calculate the weights
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  derived->launchWeightKernel(trajectory_costs_d, this->baseline_, stream);

  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs.data(),
                               trajectory_costs_d,
                               NUM_ROLLOUTS*sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR( cudaStreamSynchronize(stream) );
}
