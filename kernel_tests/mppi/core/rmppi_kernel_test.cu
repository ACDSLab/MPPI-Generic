//
// Created by mgandhi on 5/23/20.
//

#include "rmppi_kernel_test.cuh"

const int BLOCKSIZE_X = 32;
const int BLOCKSIZE_Y = 8;

template <class DYN_T, class COST_T, class FB_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelGPU(DYN_T* dynamics, COST_T* costs, FB_T* fb_controller, float dt, int num_timesteps,
                                 int optimization_stride, float lambda, float alpha, float value_func_threshold,
                                 const std::vector<float>& x0_nom, const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u, const std::vector<float>& nom_control_seq,
                                 const std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom, cudaStream_t stream)
{
  float* initial_state_d;
  float* trajectory_costs_d;
  float* control_noise_d;  // du
  float* control_std_dev_d;
  float* control_d;

  /**
   * Ensure dynamics and costs exist on GPU
   */
  dynamics->bindToStream(stream);
  costs->bindToStream(stream);
  fb_controller->bindToStream(stream);
  // Call the GPU setup functions of the model and cost
  dynamics->GPUSetup();
  costs->GPUSetup();
  fb_controller->GPUSetup();

  int control_noise_size = NUM_ROLLOUTS * num_timesteps * DYN_T::CONTROL_DIM;
  // Create x init cuda array
  HANDLE_ERROR(cudaMalloc((void**)&initial_state_d, sizeof(float) * DYN_T::STATE_DIM * 2));
  // Create control variance cuda array
  HANDLE_ERROR(cudaMalloc((void**)&control_std_dev_d, sizeof(float) * DYN_T::CONTROL_DIM));
  // create control u trajectory cuda array
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps * 2));
  // Create cost trajectory cuda array
  HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d, sizeof(float) * NUM_ROLLOUTS * 2));
  // Create zero-mean noise cuda array
  HANDLE_ERROR(
      cudaMalloc((void**)&control_noise_d, sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps * NUM_ROLLOUTS * 2));
  // Create random noise generator
  // curandGenerator_t gen;
  // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  /**
   * Fill in GPU arrays
   */
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_d, x0_act.data(), sizeof(float) * DYN_T::STATE_DIM, cudaMemcpyHostToDevice,
                               stream));
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_d + DYN_T::STATE_DIM, x0_nom.data(), sizeof(float) * DYN_T::STATE_DIM,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_std_dev_d, sigma_u.data(), sizeof(float) * DYN_T::CONTROL_DIM,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_d, nom_control_seq.data(), sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(control_d + num_timesteps * DYN_T::CONTROL_DIM, nom_control_seq.data(),
                               sizeof(float) * DYN_T::CONTROL_DIM * num_timesteps, cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_noise_d, sampled_noise.data(), sizeof(float) * control_noise_size,
                               cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(control_noise_d + control_noise_size, sampled_noise.data(),
                               sizeof(float) * control_noise_size, cudaMemcpyHostToDevice, stream));

  // curandGenerateNormal(gen, control_noise_d, control_noise_size, 0.0, 1.0);
  // HANDLE_ERROR(cudaMemcpyAsync(control_noise_d + control_noise_size,
  //                              control_noise_d,
  //                              control_noise_size * sizeof(float),
  //                              cudaMemcpyDeviceToDevice, stream));
  // Ensure copying finishes?
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  // Launch rollout kernel
  rmppi_kernels::launchRMPPIRolloutKernel<DYN_T, COST_T, typename FB_T::TEMPLATED_GPU_FEEDBACK, NUM_ROLLOUTS,
                                          BLOCKSIZE_X, BLOCKSIZE_Y, 2>(
      dynamics->model_d_, costs->cost_d_, fb_controller->getDevicePointer(), dt, num_timesteps, optimization_stride,
      lambda, alpha, value_func_threshold, initial_state_d, control_d, control_noise_d, control_std_dev_d,
      trajectory_costs_d, stream);

  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_act.data(), trajectory_costs_d, NUM_ROLLOUTS * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

  HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nom.data(), trajectory_costs_d + NUM_ROLLOUTS,
                               NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  cudaFree(initial_state_d);
  cudaFree(control_std_dev_d);
  cudaFree(control_d);
  cudaFree(trajectory_costs_d);
  cudaFree(control_noise_d);
}

template <class DYN_T, class COST_T, class FB_T, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelCPU(DYN_T* model, COST_T* costs, FB_T* fb_controller, float dt, int num_timesteps,
                                 int optimization_stride, float lambda, float alpha, float value_func_threshold,
                                 const std::vector<float>& x0_nom, const std::vector<float>& x0_act,
                                 const std::vector<float>& sigma_u, const std::vector<float>& nom_control_seq,
                                 std::vector<float>& sampled_noise,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                 std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom)
{
  // Crate Eigen items
  const int state_dim = DYN_T::STATE_DIM;
  const int control_dim = DYN_T::CONTROL_DIM;
  using control_matrix = typename COST_T::control_matrix;
  using control_array = typename DYN_T::control_array;
  using state_array = typename DYN_T::state_array;
  using output_array = typename DYN_T::output_array;
  using feedback_matrix = typename DYN_T::feedback_matrix;
  Eigen::Map<const state_array> x_init_nom(x0_nom.data());
  Eigen::Map<const state_array> x_init_act(x0_act.data());

  control_array cost_std_dev;
  for (int i = 0; i < control_dim; i++)
  {
    cost_std_dev(i) = sigma_u[i];
  }

  int control_traj_size = NUM_ROLLOUTS * num_timesteps * control_dim;

  // Start rollouts
  for (int traj_i = 0; traj_i < NUM_ROLLOUTS; traj_i++)
  {
    float cost_real_w_tracking = 0;  // S^(V, x_0, x*_0) in Grady Thesis (8.24)
    float state_cost_nom = 0;        // S(V, x*_0)
    float running_state_cost_real = 0;
    float running_control_cost_real = 0;
    int crash_status_nom[1] = { 0 };
    int crash_status_act[1] = { 0 };

    int traj_index = traj_i * num_timesteps;

    // Get all relevant values at time t in rollout i
    state_array x_t_nom = x_init_nom;
    state_array x_t_act = x_init_act;
    state_array x_next_nom = x_init_nom;
    state_array x_next_act = x_init_act;
    output_array y_nom;
    output_array y_act;

    for (int t = 0; t < num_timesteps; t++)
    {
      // Controls are read only so I can use Eigen::Map<const...>
      Eigen::Map<const control_array> u_t(nom_control_seq.data() + t * control_dim);  // trajectory u at time t
      Eigen::Map<control_array> pure_noise_act(sampled_noise.data() + (traj_index + t) * control_dim);  // Noise at time
                                                                                                        // t
      Eigen::Map<control_array> pure_noise_nom(sampled_noise.data() + control_traj_size +
                                               (traj_index + t) * control_dim);  // ptr to noise for nominal
      control_array eps_t = cost_std_dev.cwiseProduct(pure_noise_act);

      // Create newly calculated values at time t in rollout i
      state_array x_dot_t_nom;
      state_array x_dot_t_act;
      control_array u_nom;
      if (traj_i == 0 || t < optimization_stride)
      {
        eps_t = control_array::Zero();
        u_nom = u_t;
      }
      else if (traj_i >= 0.99 * NUM_ROLLOUTS)
      {
        u_nom = eps_t;
      }
      else
      {
        u_nom = u_t + eps_t;
      }

      control_array fb_u_t = fb_controller->k(x_t_act, x_t_nom, t);
      control_array u_act = u_nom + fb_u_t;

      model->enforceConstraints(x_t_nom, u_nom);
      model->enforceConstraints(x_t_act, u_act);
      float state_cost_act = 0;
      // Cost update
      if (t > 0)
      {
        control_array zero_u = control_array::Zero();
        state_cost_nom += costs->computeStateCost(y_nom, t, crash_status_nom);
        state_cost_act = costs->computeStateCost(y_act, t, crash_status_act);
        cost_real_w_tracking += state_cost_act + costs->computeFeedbackCost(fb_u_t, cost_std_dev, lambda, alpha);

        running_state_cost_real += state_cost_act;
        running_control_cost_real +=
            costs->computeLikelihoodRatioCost(u_t + fb_u_t, eps_t, cost_std_dev, lambda, alpha);
      }

      // Dyanamics Update
      model->step(x_t_nom, x_next_nom, x_dot_t_nom, u_nom, y_nom, t, dt);
      model->step(x_t_act, x_next_act, x_dot_t_act, u_act, y_act, t, dt);
      x_t_nom = x_next_nom;
      x_t_act = x_next_act;
    }

    // Compute average cost per timestep
    state_cost_nom /= ((float)num_timesteps - 1);
    cost_real_w_tracking /= ((float)num_timesteps - 1);
    running_state_cost_real /= ((float)num_timesteps - 1);

    state_cost_nom += costs->terminalCost(x_t_nom) / (num_timesteps - 1);
    cost_real_w_tracking += costs->terminalCost(x_t_act) / (num_timesteps - 1);
    running_state_cost_real += costs->terminalCost(x_t_act) / (num_timesteps - 1);

    float cost_nom =
        0.5 * state_cost_nom + 0.5 * std::max(std::min(cost_real_w_tracking, value_func_threshold), state_cost_nom);
    // Figure out control costs for the nominal trajectory
    float cost_nom_control = 0;
    for (int t = 1; t < num_timesteps; t++)
    {
      Eigen::Map<const control_array> u_nom(nom_control_seq.data() + t * control_dim);  // trajectory u at time t
      Eigen::Map<const control_array> pure_noise(sampled_noise.data() + (traj_index + t) * control_dim);  // Noise at
                                                                                                          // time t
      control_array eps_t = cost_std_dev.cwiseProduct(pure_noise);
      control_array u_t = u_nom;
      if (traj_i == 0 || t < optimization_stride)
      {
        eps_t = control_array::Zero();
      }
      else if (traj_i >= 0.99 * NUM_ROLLOUTS)
      {
        u_t = control_array::Zero();
        ;
      }
      cost_nom_control += costs->computeLikelihoodRatioCost(u_t, eps_t, cost_std_dev, lambda, alpha);
    }

    // Compute average cost per timestep
    cost_nom_control /= ((float)num_timesteps - 1);
    running_control_cost_real /= ((float)num_timesteps - 1);

    cost_nom += cost_nom_control;
    trajectory_costs_nom[traj_i] = cost_nom;
    trajectory_costs_act[traj_i] = running_state_cost_real + running_control_cost_real;
  }
}

template <class DYNAMICS_T, class COSTS_T, class FB_T, int NUM_ROLLOUTS, int NUM_TIMESTEPS, int BLOCKSIZE_X,
          int BLOCKSIZE_Y>
void launchComparisonRolloutKernelTest(
    DYNAMICS_T* dynamics, COSTS_T* costs, FB_T* fb_controller, float dt, float lambda, float alpha,
    std::array<float, DYNAMICS_T::STATE_DIM> state_array, std::array<float, DYNAMICS_T::STATE_DIM> state_array_nominal,
    std::array<float, NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM> control_array,
    std::array<float, NUM_TIMESTEPS * NUM_ROLLOUTS * DYNAMICS_T::CONTROL_DIM> control_noise_array,
    std::array<float, DYNAMICS_T::CONTROL_DIM> sigma_u, std::array<float, 2 * NUM_ROLLOUTS>& rmppi_costs_out,
    std::array<float, NUM_ROLLOUTS>& mppi_costs_out, int opt_delay, cudaStream_t stream)
{
  /*************************** MPPI ******************************************/
  float* state_d;
  float* U_d;
  float* du_d;
  float* nu_d;
  float* costs_d;

  // Allocate CUDA memory for the rollout
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * state_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&U_d, sizeof(float) * control_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&du_d, sizeof(float) * DYNAMICS_T::CONTROL_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS));
  HANDLE_ERROR(cudaMalloc((void**)&nu_d, sizeof(float) * sigma_u.size()));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * mppi_costs_out.size()));

  // Copy the initial values
  HANDLE_ERROR(
      cudaMemcpyAsync(state_d, state_array.data(), sizeof(float) * state_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(
      cudaMemcpyAsync(U_d, control_array.data(), sizeof(float) * control_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(du_d, control_noise_array.data(), sizeof(float) * control_noise_array.size(),
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(nu_d, sigma_u.data(), sizeof(float) * sigma_u.size(), cudaMemcpyHostToDevice, stream));

  const int gridsize_x = (NUM_ROLLOUTS - 1) / BLOCKSIZE_X + 1;
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  dim3 dimGrid(gridsize_x, 1, 1);
  mppi_common::rolloutKernel<DYNAMICS_T, COSTS_T, BLOCKSIZE_X, BLOCKSIZE_Y, NUM_ROLLOUTS, 1>
      <<<dimGrid, dimBlock, 0, stream>>>(dynamics->model_d_, costs->cost_d_, dt, NUM_TIMESTEPS, opt_delay, lambda,
                                         alpha, state_d, U_d, du_d, nu_d, costs_d);
  CudaCheckError();

  // Copy data back
  HANDLE_ERROR(cudaMemcpyAsync(mppi_costs_out.data(), costs_d, sizeof(float) * mppi_costs_out.size(),
                               cudaMemcpyDeviceToHost, stream));

  HANDLE_ERROR(cudaStreamSynchronize(stream));
  // Deallocate CUDA Memory
  HANDLE_ERROR(cudaFree(state_d));
  HANDLE_ERROR(cudaFree(U_d));
  HANDLE_ERROR(cudaFree(du_d));
  HANDLE_ERROR(cudaFree(nu_d));
  HANDLE_ERROR(cudaFree(costs_d));

  /*************************** RMPPI ******************************************/

  // Allocate CUDA memory for the rollout
  HANDLE_ERROR(cudaMalloc((void**)&state_d, sizeof(float) * 2 * state_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&U_d, sizeof(float) * 2 * control_array.size()));
  HANDLE_ERROR(cudaMalloc((void**)&du_d, sizeof(float) * 2 * DYNAMICS_T::CONTROL_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS));
  HANDLE_ERROR(cudaMalloc((void**)&nu_d, sizeof(float) * sigma_u.size()));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * rmppi_costs_out.size()));

  // Copy the initial values
  HANDLE_ERROR(
      cudaMemcpyAsync(state_d, state_array.data(), sizeof(float) * state_array.size(), cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(state_d + DYNAMICS_T::STATE_DIM, state_array_nominal.data(),
                               sizeof(float) * DYNAMICS_T::STATE_DIM, cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(nu_d, sigma_u.data(), sizeof(float) * sigma_u.size(), cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaMemcpyAsync(U_d, control_array.data(), sizeof(float) * DYNAMICS_T::CONTROL_DIM * NUM_TIMESTEPS,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(U_d + NUM_TIMESTEPS * DYNAMICS_T::CONTROL_DIM, control_array.data(),
                               sizeof(float) * DYNAMICS_T::CONTROL_DIM * NUM_TIMESTEPS, cudaMemcpyHostToDevice,
                               stream));

  HANDLE_ERROR(cudaMemcpyAsync(du_d, control_noise_array.data(), sizeof(float) * control_noise_array.size(),
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(du_d + control_noise_array.size(), control_noise_array.data(),
                               sizeof(float) * control_noise_array.size(), cudaMemcpyHostToDevice, stream));

  dimBlock = dim3(BLOCKSIZE_X, BLOCKSIZE_Y, 2);
  dimGrid = dim3(gridsize_x, 1, 1);

  rmppi_kernels::RMPPIRolloutKernel<DYNAMICS_T, COSTS_T, typename FB_T::TEMPLATED_GPU_FEEDBACK, BLOCKSIZE_X,
                                    BLOCKSIZE_Y, NUM_ROLLOUTS, 2><<<dimGrid, dimBlock, 0, stream>>>(
      dynamics->model_d_, costs->cost_d_, fb_controller->getDevicePointer(), dt, NUM_TIMESTEPS, opt_delay, lambda,
      alpha, 10, state_d, U_d, du_d, nu_d, costs_d);

  // Copy data back
  HANDLE_ERROR(cudaMemcpyAsync(rmppi_costs_out.data(), costs_d, sizeof(float) * rmppi_costs_out.size(),
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  // Deallocate CUDA Memory
  HANDLE_ERROR(cudaFree(state_d));
  HANDLE_ERROR(cudaFree(U_d));
  HANDLE_ERROR(cudaFree(du_d));
  HANDLE_ERROR(cudaFree(nu_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

template <class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void launchRMPPIRolloutKernelCCMCPU(DYN_T* model, COST_T* costs, ccm::LinearCCM<DYN_T, NUM_TIMESTEPS>* fb_controller,
                                    float dt, int num_timesteps, int optimization_stride, float lambda, float alpha,
                                    float value_func_threshold, const std::vector<float>& x0_nom,
                                    const std::vector<float>& x0_act, const std::vector<float>& sigma_u,
                                    const std::vector<float>& nom_control_seq, std::vector<float>& sampled_noise,
                                    std::array<float, NUM_ROLLOUTS>& trajectory_costs_act,
                                    std::array<float, NUM_ROLLOUTS>& trajectory_costs_nom)
{
  // Crate Eigen items
  const int state_dim = DYN_T::STATE_DIM;
  const int control_dim = DYN_T::CONTROL_DIM;
  using control_matrix = typename COST_T::control_matrix;
  using control_array = typename DYN_T::control_array;
  using state_array = typename DYN_T::state_array;
  using output_array = typename DYN_T::output_array;
  using feedback_matrix = typename DYN_T::feedback_matrix;
  using dfdx = typename DYN_T::dfdx;
  Eigen::Map<const state_array> x_init_nom(x0_nom.data());
  Eigen::Map<const state_array> x_init_act(x0_act.data());

  // CCM Initialization
  // ccm::Vectorf<7> pts, weights;
  // std::tie(pts, weights) = ccm::chebyshevPts<7>();
  // auto CCM_Controller = ccm::LinearCCM<DYN_T>(model);
  // dfdx M_new = dfdx::Identity();
  // M_new(3,3) = 0.01;
  // M_new(2,2) = 0.01;
  // CCM_Controller.setM(M_new);

  control_array cost_std_dev;
  for (int i = 0; i < control_dim; i++)
  {
    cost_std_dev(i) = sigma_u[i];
  }

  int control_traj_size = NUM_ROLLOUTS * num_timesteps * control_dim;

  // Start rollouts
  for (int traj_i = 0; traj_i < NUM_ROLLOUTS; traj_i++)
  {
    float cost_real_w_tracking = 0;  // S^(V, x_0, x*_0) in Grady Thesis (8.24)
    float state_cost_nom = 0;        // S(V, x*_0)
    float running_state_cost_real = 0;
    float running_control_cost_real = 0;
    float running_control_cost_nom = 0;

    int traj_index = traj_i * num_timesteps;

    // Get all relevant values at time t in rollout i
    state_array x_t_nom = x_init_nom;
    state_array x_t_act = x_init_act;
    state_array x_next_nom = x_init_nom;
    state_array x_next_act = x_init_act;
    output_array y_nom;
    output_array y_act;

    for (int t = 0; t < num_timesteps; t++)
    {
      // Controls are read only so I can use Eigen::Map<const...>
      Eigen::Map<const control_array> u_t(nom_control_seq.data() + t * control_dim);  // trajectory u at time t
      /**
       * Get the noise at time t for nominal and actual systems
       * Note that they are assumed to be the same
       * They both need to exist to save different controls to
       * later on.
       */
      Eigen::Map<control_array> pure_noise_act(sampled_noise.data() + (traj_index + t) * control_dim);  // Noise at time
                                                                                                        // t
      Eigen::Map<control_array> pure_noise_nom(sampled_noise.data() + control_traj_size +
                                               (traj_index + t) * control_dim);  // ptr to noise for nominal
      control_array eps_t = cost_std_dev.cwiseProduct(pure_noise_act);
      // Eigen::Map<const feedback_matrix>
      //     feedback_gains_t(feedback_gains_seq.data() + t * control_dim * state_dim); // Feedback gains at time t

      // Create newly calculated values at time t in rollout i
      state_array x_dot_t_nom;
      state_array x_dot_t_act;
      control_array u_nom;
      if (traj_i == 0 || t < optimization_stride)
      {
        eps_t = control_array::Zero();
        u_nom = u_t;
      }
      else if (traj_i >= 0.99 * NUM_ROLLOUTS)
      {
        u_nom = eps_t;
      }
      else
      {
        u_nom = u_t + eps_t;
      }
      bool debug = false;
      // if (traj_i == 0) {
      //   debug = true;
      // }
      // control_array fb_u_t = feedback_gains_t * (x_t_act - x_t_nom);
      control_array fb_u_t = fb_controller->u_feedback(x_t_act, x_t_nom, u_nom, debug);
      if (traj_i == -1)
      {
        std::cout << "Feedback at t = " << t << ": " << fb_u_t.transpose() << std::endl;
        std::cout << "\tu_nominl: " << u_nom.transpose() << std::endl;
        std::cout << "\tx_actual: " << x_t_act.transpose() << std::endl;
        std::cout << "\tx_nominl: " << x_t_nom.transpose() << std::endl;
        std::cout << std::endl;
      }

      control_array u_act = u_nom + fb_u_t;

      model->enforceConstraints(x_t_nom, u_nom);
      model->enforceConstraints(x_t_act, u_act);

      /**
       * Copy controls back into noise vecotrs
       * This is where the noise pointing to different locations
       * for nominal and actual matter.
       */
      pure_noise_act = u_act;
      pure_noise_nom = u_nom;

      // Cost update
      if (t > 0)
      {
        state_cost_nom += costs->computeStateCost(y_nom);
        float state_cost_act = costs->computeStateCost(y_act);
        cost_real_w_tracking += state_cost_act + costs->computeFeedbackCost(fb_u_t, cost_std_dev, lambda, alpha);

        running_state_cost_real += state_cost_act;
        running_control_cost_real +=
            costs->computeLikelihoodRatioCost(u_nom - eps_t + fb_u_t, eps_t, cost_std_dev, lambda, alpha);
        running_control_cost_nom +=
            costs->computeLikelihoodRatioCost(u_nom - eps_t, eps_t, cost_std_dev, lambda, alpha);
      }

      // Dyanamics Update
      model->step(x_t_nom, x_next_nom, x_dot_t_nom, u_nom, y_nom, t, dt);
      model->step(x_t_act, x_next_act, x_dot_t_act, u_act, y_act, t, dt);
      x_t_nom = x_next_nom;
      x_t_act = x_next_act;
    }

    // Compute average cost per timestep
    state_cost_nom /= ((float)num_timesteps - 1);
    cost_real_w_tracking /= ((float)num_timesteps - 1);
    running_state_cost_real /= ((float)num_timesteps - 1);

    state_cost_nom += costs->terminalCost(y_nom) / (num_timesteps - 1);
    cost_real_w_tracking += costs->terminalCost(y_act) / (num_timesteps - 1);
    running_state_cost_real += costs->terminalCost(y_act) / (num_timesteps - 1);

    float cost_nom =
        0.5 * state_cost_nom + 0.5 * std::max(std::min(cost_real_w_tracking, value_func_threshold), state_cost_nom);
    // Figure out control costs for the nominal trajectory
    // float cost_nom_control = 0;
    // for (int t = 0; t < num_timesteps - 1; t++) {
    //   Eigen::Map<const control_array>
    //       u_nom(nom_control_seq.data() + t * control_dim); // trajectory u at time t
    //   Eigen::Map<const control_array>
    //       eps_t(sampled_noise.data() + control_traj_size +
    //                  (traj_index + t) * control_dim); // U + noise Noise at time t
    //   // control_array eps_t = cost_std_dev.cwiseProduct(pure_noise);
    //   // control_array u_t = u_nom;
    //   // if (traj_i == 0) {
    //   //   eps_t = control_array::Zero();
    //   // } else if (traj_i >= 0.99 * NUM_ROLLOUTS) {
    //   //   u_t = control_array::Zero();;
    //   // }
    //   cost_nom_control += costs->computeLikelihoodRatioCost(u_nom, eps_t - u_nom, cost_std_dev,
    //                                                         lambda, alpha);
    // }
    // Compute average cost per timestep

    running_control_cost_nom /= (float)(num_timesteps - 1);
    running_control_cost_real /= (float)(num_timesteps - 1);

    cost_nom += running_control_cost_nom;
    trajectory_costs_nom[traj_i] = cost_nom;
    trajectory_costs_act[traj_i] = running_state_cost_real + running_control_cost_real;
  }
}
