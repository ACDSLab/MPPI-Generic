//
// Created by mgandhi on 5/23/20.
//

#include "rmppi_kernel_test.cuh"

const int BLOCKSIZE_X = 32;
const int BLOCKSIZE_Y = 8;

template<class DYN_T, class COST_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelGPU(DYN_T* dynamics, COST_T* costs,
                                 float dt,
                                 int num_timesteps,
                                 float lambda,
                                 float alpha,
                                 float value_func_threshold,
                                 const std::vector<float>& x0_nom,
                                 const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u,
                                 const std::vector<float>& nom_control_seq,
                                 const std::vector<float>& feedback_gains_seq,
                                 const std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom,
                                 cudaStream_t stream) {
  float* initial_state_d;
  float* trajectory_costs_d;
  float* control_noise_d; // du
  float* control_std_dev_d;
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
  HANDLE_ERROR(cudaMalloc((void**)&control_std_dev_d,
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
  // curandGenerator_t gen;
  // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  /**
   * Fill in GPU arrays
   */
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_d, x0_act.data(),
                               sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_d + DYN_T::STATE_DIM, x0_nom.data(),
                               sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_std_dev_d, sigma_u.data(),
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

  HANDLE_ERROR(cudaMemcpyAsync(control_noise_d,
                               sampled_noise.data(),
                               sizeof(float) * control_noise_size,
                               cudaMemcpyHostToDevice, stream));

 HANDLE_ERROR(cudaMemcpyAsync(control_noise_d + control_noise_size,
                               sampled_noise.data(),
                               sizeof(float) * control_noise_size,
                               cudaMemcpyHostToDevice, stream));

  // curandGenerateNormal(gen, control_noise_d, control_noise_size, 0.0, 1.0);
  // HANDLE_ERROR(cudaMemcpyAsync(control_noise_d + control_noise_size,
  //                              control_noise_d,
  //                              control_noise_size * sizeof(float),
  //                              cudaMemcpyDeviceToDevice, stream));
  // Ensure copying finishes?
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  // Launch rollout kernel
  rmppi_kernels::launchRMPPIRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BLOCKSIZE_X,
    BLOCKSIZE_Y, 2>(dynamics->model_d_, costs->cost_d_, dt, num_timesteps,
                    lambda, alpha, value_func_threshold, initial_state_d, control_d,
                    control_noise_d, feedback_gains_d, control_std_dev_d,
                    trajectory_costs_d, stream);


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
  cudaFree(control_std_dev_d);
  cudaFree(control_d);
  cudaFree(trajectory_costs_d);
  cudaFree(feedback_gains_d);
  cudaFree(control_noise_d);
}

template<class DYN_T, class COST_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelCPU(DYN_T* model, COST_T* costs,
                                 float dt,
                                 int num_timesteps,
                                 float lambda,
                                 float alpha,
                                 float value_func_threshold,
                                 const std::vector<float>& x0_nom,
                                 const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u,
                                 const std::vector<float>& nom_control_seq,
                                 const std::vector<float>& feedback_gains_seq,
                                 const std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom) {
  // Crate Eigen items
  const int state_dim = DYN_T::STATE_DIM;
  const int control_dim = DYN_T::CONTROL_DIM;
  using control_matrix = typename COST_T::control_matrix;
  using control_array = typename DYN_T::control_array;
  using state_array = typename DYN_T::state_array;
  using feedback_matrix = typename DYN_T::feedback_matrix;
  Eigen::Map<const state_array> x_init_nom(x0_nom.data());
  Eigen::Map<const state_array> x_init_act(x0_act.data());

  control_array cost_std_dev;
  for(int i = 0; i < control_dim; i++) {
    cost_std_dev(i) = sigma_u[i];
  }

  // Start rollouts
  for (int traj_i = 0; traj_i < NUM_ROLLOUTS; traj_i++)  {
    float cost_real_w_tracking = 0; // S^(V, x_0, x*_0) in Grady Thesis (8.24)
    float state_cost_nom = 0; // S(V, x*_0)
    float running_state_cost_real = 0;
    float running_control_cost_real = 0;

    int traj_index = traj_i * num_timesteps;

    // Get all relevant values at time t in rollout i
    state_array x_t_nom = x_init_nom;
    state_array x_t_act = x_init_act;

    for (int t = 0; t < num_timesteps; t++){
      // Controls are read only so I can use Eigen::Map<const...>
      Eigen::Map<const control_array>
          u_t(nom_control_seq.data() + t * control_dim); // trajectory u at time t
      Eigen::Map<const control_array>
          pure_noise(sampled_noise.data() + (traj_index + t) * control_dim); // Noise at time t
      control_array eps_t = cost_std_dev.cwiseProduct(pure_noise);
      Eigen::Map<const feedback_matrix>
          feedback_gains_t(feedback_gains_seq.data() + t * control_dim * state_dim); // Feedback gains at time t

      // Create newly calculated values at time t in rollout i
      state_array x_dot_t_nom;
      state_array x_dot_t_act;
      control_array u_nom;
      if (traj_i == 0) {
        eps_t = control_array::Zero();
        u_nom = u_t;
      } else if (traj_i >= 0.99 * NUM_ROLLOUTS) {
        u_nom = eps_t;
      } else {
         u_nom = u_t + eps_t;
      }


      control_array fb_u_t = feedback_gains_t * (x_t_act - x_t_nom);
      control_array u_act = u_nom + fb_u_t;

      // Cost update
      control_array zero_u = control_array::Zero();
      state_cost_nom += costs->computeStateCost(x_t_nom)*dt;
      float state_cost_act = costs->computeStateCost(x_t_act)*dt;
      cost_real_w_tracking += state_cost_act +
                              costs->computeFeedbackCost(fb_u_t, cost_std_dev, lambda, alpha)*dt;

      running_state_cost_real += state_cost_act;
      running_control_cost_real +=
      costs->computeLikelihoodRatioCost(u_t + fb_u_t, eps_t, cost_std_dev, lambda, alpha)*dt;

      model->enforceConstraints(x_t_nom, u_nom);
      model->enforceConstraints(x_t_act, u_act);

      // Dyanamics Update
      model->computeStateDeriv(x_t_nom, u_nom, x_dot_t_nom);
      model->computeStateDeriv(x_t_act, u_act, x_dot_t_act);

      model->updateState(x_t_act, x_dot_t_act, dt);
      model->updateState(x_t_nom, x_dot_t_nom, dt);
    }

    state_cost_nom += costs->terminalCost(x_t_nom);
    cost_real_w_tracking += costs->terminalCost(x_t_act);
    running_state_cost_real += costs->terminalCost(x_t_act);

    float cost_nom = 0.5 * state_cost_nom + 0.5 *
      std::max(std::min(cost_real_w_tracking, value_func_threshold), state_cost_nom);
    // Figure out control costs for the nominal trajectory
    float cost_nom_control = 0;
    for (int t = 0; t < num_timesteps - 1; t++) {
      Eigen::Map<const control_array>
          u_nom(nom_control_seq.data() + t * control_dim); // trajectory u at time t
      Eigen::Map<const control_array>
          pure_noise(sampled_noise.data() + (traj_index + t) * control_dim); // Noise at time t
      control_array eps_t = cost_std_dev.cwiseProduct(pure_noise);
      control_array u_t = u_nom;
      if (traj_i == 0) {
        eps_t = control_array::Zero();
      } else if (traj_i >= 0.99 * NUM_ROLLOUTS) {
        u_t = control_array::Zero();;
      }
      cost_nom_control += costs->computeLikelihoodRatioCost(u_t, eps_t, cost_std_dev, lambda, alpha)*dt;
    }

    cost_nom += cost_nom_control;
    trajectory_costs_nom[traj_i] = cost_nom;
    trajectory_costs_act[traj_i] = running_state_cost_real + running_control_cost_real;
  }
}
