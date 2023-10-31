//
// Created by mgandhi on 5/23/20.
//

#include "rmppi_kernel_test.cuh"

template <class DYN_T, class COST_T, class SAMPLER_T>
void launchCPUInitEvalKernel(DYN_T* model, COST_T* cost, SAMPLER_T* sampler, const float dt, const int num_timesteps,
                             const int num_candidates, const int num_samples, const float lambda, const float alpha,
                             const Eigen::Ref<const Eigen::MatrixXf>& candidates,
                             const Eigen::Ref<const Eigen::MatrixXi>& strides,
                             Eigen::Ref<Eigen::MatrixXf> trajectory_costs)
{
  using state_array = typename DYN_T::state_array;
  using output_array = typename DYN_T::output_array;
  using control_array = typename DYN_T::control_array;
  const int num_rollouts = num_candidates * num_samples;
  Eigen::MatrixXi crash_status = Eigen::MatrixXi::Zero(num_rollouts, 1);

  // Get control samples from sampler
  Eigen::MatrixXf control_noise = Eigen::MatrixXf::Zero(DYN_T::CONTROL_DIM, num_rollouts * num_timesteps);
  HANDLE_ERROR(cudaMemcpy(control_noise.data(), sampler->getControlSample(0, 0, 0),
                          sizeof(float) * DYN_T::CONTROL_DIM * num_rollouts * num_timesteps, cudaMemcpyDeviceToHost));

  state_array curr_state, next_state, state_der;
  output_array output;
  control_array u;
  for (int candidate_idx = 0; candidate_idx < num_candidates; candidate_idx++)
  {
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++)
    {
      int global_idx = candidate_idx * num_samples + sample_idx;
      float& running_cost = trajectory_costs(global_idx, 0);
      curr_state = candidates.col(candidate_idx);
      model->initializeDynamics(curr_state, u, output, 0, dt);
      cost->initializeCosts(output, u, 0, dt);
      // sampler->initializeDistributions();
      for (int t = 0; t < num_timesteps; t++)
      {
        int control_index = sample_idx * num_timesteps + min(t + strides(candidate_idx, 0), num_timesteps - 1);
        u = control_noise.col(control_index);
        model->enforceConstraints(curr_state, u);
        model->step(curr_state, next_state, state_der, u, output, t, dt);
        // if (t > 0)
        // {
        running_cost += cost->computeRunningCost(output, u, t, &crash_status(global_idx));
        running_cost += sampler->computeLikelihoodRatioCost(u, t, 0, lambda, alpha);
        // }
        curr_state = next_state;
      }
      running_cost += cost->terminalCost(output);
      running_cost /= (num_timesteps);
    }
  }
}
