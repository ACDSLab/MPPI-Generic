//
// Created by mgandhi on 5/23/20.
//

#include "rmppi_kernel_test.cuh"

const int BLOCKSIZE_X = 32;
const int BLOCKSIZE_Y = 8;

template<class DYN_T, class COST_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernel(DYN_T* dynamics, COST_T* costs,
                              float dt,
                              int num_timesteps,
                              float lambda,
                              const std::vector<float>& x0,
                              const std::vector<float>& sigma_u,
                              const std::vector<float>& nom_control_seq,
                              const std::vector<float>& feedback_gains_seq,
                              std::vector<float>& trajectory_costs_act,
                              std::vector<float>& trajectory_costs_nom,
                              cudaStream_t stream) {
  float* initial_state_d;
  float* trajectory_costs_d;
  float* control_noise_d; // du
  float* control_variance_d;
  float* control_d;
  float* feedback_gains_d;

  /**
   * Ensure dynamics and costs exist on GPU
   */
  dynamics->bindToStream(stream);
  costs->bindToStream(stream);
  // Call the GPU setup functions of the model and cost
  dynamics->GPUSetup();
  costs->GPUSetup();

  int control_noise_size = NUM_ROLLOUTS * num_timesteps * DYN_T::CONTROL_DIM;
  // Create x init cuda array
  HANDLE_ERROR(cudaMalloc((void**)&initial_state_d,
                          sizeof(float) * DYN_T::STATE_DIM * 2));
  // Create control variance cuda array
  HANDLE_ERROR(cudaMalloc((void**)&control_variance_d,
                          sizeof(float) * DYN_T::CONTROL_DIM));
  // create control u trajectory cuda array
  HANDLE_ERROR(cudaMalloc((void**)&control_d,
                          sizeof(float) * DYN_T::CONTROL_DIM *
                          num_timesteps * 2));
  // Create cost trajectory cuda array
  HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d,
                          sizeof(float) * NUM_ROLLOUTS * 2));
  // Create zero-mean noise cuda array
  HANDLE_ERROR(cudaMalloc((void**)&control_noise_d,
                          sizeof(float) * DYN_T::CONTROL_DIM *
                          num_timesteps * NUM_ROLLOUTS * 2));
  // Create feedback_gains sequence array
  HANDLE_ERROR(cudaMalloc((void**)&feedback_gains_d,
                          sizeof(float) * DYN_T::CONTROL_DIM *
                          DYN_T::STATE_DIM * num_timesteps));
  // Create random noise generator
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  /**
   * Fill in GPU arrays
   */
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_d, x0.data(),
                               sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_d + DYN_T::STATE_DIM, x0.data(),
                               sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_variance_d, sigma_u.data(),
                               sizeof(float) * DYN_T::CONTROL_DIM,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_d, nom_control_seq.data(),
                               sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(control_d + num_timesteps * DYN_T::CONTROL_DIM,
                               nom_control_seq.data(),
                               sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(feedback_gains_d,
                               feedback_gains_seq.data(),
                               sizeof(float) * DYN_T::CONTROL_DIM *
                               DYN_T::STATE_DIM * num_timesteps,
                               cudaMemcpyHostToDevice, stream));

  curandGenerateNormal(gen, control_noise_d, control_noise_size, 0.0, 1.0);
  HANDLE_ERROR(cudaMemcpyAsync(control_noise_d + control_noise_size,
                               control_noise_d,
                               control_noise_size * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
  // Ensure copying finishes?
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  // Launch rollout kernel
  rmppi_kernels::launchRMPPIRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BLOCKSIZE_X,
    BLOCKSIZE_Y, 2>(dynamics->model_d_, costs->cost_d_, dt, num_timesteps,
                    initial_state_d, control_d, control_noise_d,
                    feedback_gains_d, control_variance_d,
                    trajectory_costs_d, lambda, stream);

  // Copy the costs back to the host
  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_act.data(),
                               trajectory_costs_d,
                               NUM_ROLLOUTS * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nom.data(),
                               trajectory_costs_d + NUM_ROLLOUTS,
                               NUM_ROLLOUTS * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  cudaFree(initial_state_d);
  cudaFree(control_variance_d);
  cudaFree(control_d);
  cudaFree(trajectory_costs_d);
  cudaFree(feedback_gains_d);
  cudaFree(control_noise_d);
}
