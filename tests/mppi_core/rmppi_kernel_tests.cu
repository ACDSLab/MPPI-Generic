//
// Created by mgandhi on 5/23/20.
//
#include <gtest/gtest.h>
#include <kernel_tests/core/rmppi_kernel_test.cuh>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/core/rmppi_kernels.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/utils/test_helper.h>

#include <iostream>
#include <vector>

class RMPPIKernels : public ::testing::Test
{
public:
  static const int num_timesteps = 50;
  using DYN_T = DoubleIntegratorDynamics;
  using COST_T = DoubleIntegratorCircleCost;
  // using COST_T = QuadraticCost<DYN_T>;
  using SAMPLER_T = mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>;
  using FB_T = DDPFeedback<DYN_T, num_timesteps>;
  using control_trajectory = Eigen::Matrix<float, DYN_T::CONTROL_DIM, num_timesteps>;
  using state_array = DYN_T::state_array;
  using output_array = DYN_T::output_array;
  using control_array = DYN_T::control_array;

  float dt = 0.01;
  float lambda = 1.1f;
  float alpha = 0.0;
  float std_dev = 2.0f;
  curandGenerator_t gen;
  cudaStream_t stream;

  void SetUp() override
  {
    model = new DYN_T(10);  // Initialize the double integrator DYN_T
    cost = new COST_T;      // Initialize the cost function
    auto sampler_params = SAMPLER_T::SAMPLING_PARAMS_T();
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      sampler_params.std_dev[i] = std_dev;
    }
    sampler = new SAMPLER_T(sampler_params);
    fb_controller = new FB_T(model, dt);

    HANDLE_CURAND_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen, seed));

    HANDLE_ERROR(cudaStreamCreate(&stream));
    model->bindToStream(stream);
    cost->bindToStream(stream);
    sampler->bindToStream(stream);
    fb_controller->bindToStream(stream);
    curandSetStream(gen, stream);

    model->GPUSetup();
    cost->GPUSetup();
    sampler->GPUSetup();
    fb_controller->GPUSetup();
  }

  void TearDown() override
  {
    delete model;
    delete cost;
    delete sampler;
  }

  DYN_T* model;
  COST_T* cost;
  SAMPLER_T* sampler;
  FB_T* fb_controller;
};

/**
 * @brief Runs the combined init eval kernel and compares to one created on the CPU
 * to ensure they produce the same cost trajectories. Multiple runs of the init eval kernel
 * will be done with different parallelizations to ensure that the kernel calculates the same
 * regardless of the parallelization technique.
 */
TEST_F(RMPPIKernels, ValidateCombinedInitEvalKernel)
{
  /**
   * Set up the problem
   */
  int num_candidates = 9;
  int num_samples = 64;
  int num_rollouts = num_candidates * num_samples;

  /**
   * Setup GPU
   */
  int* strides_d;
  float* initial_x_d;
  float* cost_trajectories_d;
  sampler->setNumTimesteps(num_timesteps);
  sampler->setNumRollouts(num_rollouts);
  HANDLE_ERROR(cudaMalloc((void**)&strides_d, sizeof(int) * num_candidates));
  HANDLE_ERROR(cudaMalloc((void**)&initial_x_d, sizeof(float) * num_candidates * DYN_T::STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&cost_trajectories_d, sizeof(float) * num_rollouts));

  // Setup inputs on both CPU and GPU
  Eigen::MatrixXf candidates = Eigen::MatrixXf::Random(DYN_T::STATE_DIM, num_candidates);
  Eigen::MatrixXi strides = Eigen::MatrixXi::Zero(num_candidates, 1);
  for (int i = 0; i < num_candidates; i++)
  {
    strides(i) = i + 1;
  }
  Eigen::MatrixXf trajectory_costs_cpu = Eigen::MatrixXf::Zero(num_rollouts, 1);
  Eigen::MatrixXf trajectory_costs_gpu = Eigen::MatrixXf::Zero(num_rollouts, 1);

  control_trajectory nominal_trajectory = control_trajectory::Random();
  sampler->copyImportanceSamplerToDevice(nominal_trajectory.data(), 0, false);
  sampler->generateSamples(1, 0, gen, false);
  HANDLE_ERROR(cudaMemcpyAsync(initial_x_d, candidates.data(), sizeof(float) * DYN_T::STATE_DIM * num_candidates,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(
      cudaMemcpyAsync(strides_d, strides.data(), sizeof(int) * num_candidates, cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  /**
   * Do CPU calculation
   */
  launchCPUInitEvalKernel<DYN_T, COST_T, SAMPLER_T>(model, cost, sampler, dt, num_timesteps, num_candidates,
                                                    num_samples, lambda, alpha, candidates, strides,
                                                    trajectory_costs_cpu);

  /**
   * Do GPU Calculation on various thread dimensions
   */
  std::vector<int> possible_thread_x;
  for (int size = num_samples; size > 0; size /= 2)
  {
    possible_thread_x.push_back(size);
  }
  std::vector<int> possible_thread_y{ 1, 2, 3, 8 };
  for (const auto& thread_x : possible_thread_x)
  {
    for (const auto& thread_y : possible_thread_y)
    {
      dim3 threadsPerBlock(thread_x, thread_y, 1);
      mppi::kernels::rmppi::launchInitEvalKernel<DYN_T, COST_T, SAMPLER_T>(
          model->model_d_, cost->cost_d_, sampler->sampling_d_, dt, num_timesteps, num_rollouts, lambda, alpha,
          num_samples, strides_d, initial_x_d, cost_trajectories_d, threadsPerBlock, stream, false);
      HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_gpu.data(), cost_trajectories_d, sizeof(float) * num_rollouts,
                                   cudaMemcpyDeviceToHost, stream));
      HANDLE_ERROR(cudaStreamSynchronize(stream));
      for (int i = 0; i < num_rollouts; i++)
      {
        float cost_diff = trajectory_costs_cpu(i) - trajectory_costs_gpu(i);
        std::string error_prefix = "Eval sample " + std::to_string(i) + " (" + std::to_string(threadsPerBlock.x) +
                                   ", " + std::to_string(threadsPerBlock.y) + ", " + std::to_string(threadsPerBlock.z) +
                                   ")";
        EXPECT_LT(fabsf(cost_diff), 1e-3 * trajectory_costs_cpu(i))
            << error_prefix << ": CPU = " << trajectory_costs_cpu(i) << ", GPU = " << trajectory_costs_gpu(i)
            << std::endl;
      }
    }
  }
}

TEST_F(RMPPIKernels, ValidateSplitInitEvalKernel)
{
  /**
   * Set up the problem
   */
  int num_candidates = 9;
  int num_samples = 16;
  int num_rollouts = num_candidates * num_samples;

  /**
   * Setup GPU
   */
  int* strides_d;
  float* initial_x_d;
  float* cost_trajectories_d;
  float* output_d;
  sampler->setNumTimesteps(num_timesteps);
  sampler->setNumRollouts(num_rollouts);
  HANDLE_ERROR(cudaMalloc((void**)&strides_d, sizeof(int) * num_candidates));
  HANDLE_ERROR(cudaMalloc((void**)&initial_x_d, sizeof(float) * num_candidates * DYN_T::STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&cost_trajectories_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&output_d, sizeof(float) * num_rollouts * num_timesteps * DYN_T::OUTPUT_DIM));

  // Setup inputs on both CPU and GPU
  Eigen::MatrixXf candidates = Eigen::MatrixXf::Random(DYN_T::STATE_DIM, num_candidates) * 4;
  Eigen::MatrixXi strides = Eigen::MatrixXi::Zero(num_candidates, 1);
  for (int i = 0; i < num_candidates; i++)
  {
    strides(i) = i + 1;
  }
  Eigen::MatrixXf trajectory_costs_cpu = Eigen::MatrixXf::Zero(num_rollouts, 1);
  Eigen::MatrixXf trajectory_costs_gpu = Eigen::MatrixXf::Zero(num_rollouts, 1);

  control_trajectory nominal_trajectory = control_trajectory::Random();
  sampler->copyImportanceSamplerToDevice(nominal_trajectory.data(), 0, false);
  sampler->generateSamples(1, 0, gen, false);
  HANDLE_ERROR(cudaMemcpyAsync(initial_x_d, candidates.data(), sizeof(float) * DYN_T::STATE_DIM * num_candidates,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(
      cudaMemcpyAsync(strides_d, strides.data(), sizeof(int) * num_candidates, cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  /**
   * Do CPU calculation
   */
  launchCPUInitEvalKernel<DYN_T, COST_T, SAMPLER_T>(model, cost, sampler, dt, num_timesteps, num_candidates,
                                                    num_samples, lambda, alpha, candidates, strides,
                                                    trajectory_costs_cpu);

  /**
   * Do GPU Calculation on various thread dimensions
   */
  std::vector<int> possible_dyn_thread_x;
  for (int size = num_samples; size > 0; size /= 2)
  {
    possible_dyn_thread_x.push_back(size);
  }
  std::vector<int> possible_cost_thread_x;
  for (int size = num_timesteps - 1; size > 0; size /= 2)
  {
    possible_cost_thread_x.push_back(size);
  }
  std::vector<int> possible_thread_y{ 1, 2, 3, 8 };
  for (const auto& dyn_thread_x : possible_dyn_thread_x)
  {
    for (const auto& dyn_thread_y : possible_thread_y)
    {
      dim3 dynThreadsPerBlock(dyn_thread_x, dyn_thread_y, 1);
      for (const auto& cost_thread_x : possible_cost_thread_x)
      {
        for (const auto& cost_thread_y : possible_thread_y)
        {
          dim3 costThreadsPerBlock(cost_thread_x, cost_thread_y, 1);
          mppi::kernels::rmppi::launchSplitInitEvalKernel<DYN_T, COST_T, SAMPLER_T, true>(
              model->model_d_, cost->cost_d_, sampler->sampling_d_, dt, num_timesteps, num_rollouts, lambda, alpha,
              num_samples, strides_d, initial_x_d, output_d, cost_trajectories_d, dynThreadsPerBlock,
              costThreadsPerBlock, stream, false);
          HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_gpu.data(), cost_trajectories_d, sizeof(float) * num_rollouts,
                                       cudaMemcpyDeviceToHost, stream));
          HANDLE_ERROR(cudaStreamSynchronize(stream));
          for (int i = 0; i < num_rollouts; i++)
          {
            float cost_diff = trajectory_costs_cpu(i) - trajectory_costs_gpu(i);
            std::string error_prefix =
                "Coalesced eval sample " + std::to_string(i) + "/" + std::to_string(num_rollouts) + " dyn(" +
                std::to_string(dynThreadsPerBlock.x) + ", " + std::to_string(dynThreadsPerBlock.y) + ", " +
                std::to_string(dynThreadsPerBlock.z) + ") cost(" + std::to_string(costThreadsPerBlock.x) + ", " +
                std::to_string(costThreadsPerBlock.y) + ", " + std::to_string(costThreadsPerBlock.z) + ")";
            ASSERT_LT(fabsf(cost_diff), 1e-3 * trajectory_costs_cpu(i))
                << error_prefix << ": CPU = " << trajectory_costs_cpu(i) << ", GPU = " << trajectory_costs_gpu(i)
                << std::endl;
          }
          mppi::kernels::rmppi::launchSplitInitEvalKernel<DYN_T, COST_T, SAMPLER_T, false>(
              model->model_d_, cost->cost_d_, sampler->sampling_d_, dt, num_timesteps, num_rollouts, lambda, alpha,
              num_samples, strides_d, initial_x_d, output_d, cost_trajectories_d, dynThreadsPerBlock,
              costThreadsPerBlock, stream, false);
          HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_gpu.data(), cost_trajectories_d, sizeof(float) * num_rollouts,
                                       cudaMemcpyDeviceToHost, stream));
          HANDLE_ERROR(cudaStreamSynchronize(stream));
          for (int i = 0; i < num_rollouts; i++)
          {
            float cost_diff = trajectory_costs_cpu(i) - trajectory_costs_gpu(i);
            std::string error_prefix =
                "Non-coalesced eval sample " + std::to_string(i) + "/" + std::to_string(num_rollouts) + " dyn(" +
                std::to_string(dynThreadsPerBlock.x) + ", " + std::to_string(dynThreadsPerBlock.y) + ", " +
                std::to_string(dynThreadsPerBlock.z) + ") cost(" + std::to_string(costThreadsPerBlock.x) + ", " +
                std::to_string(costThreadsPerBlock.y) + ", " + std::to_string(costThreadsPerBlock.z) + ")";
            ASSERT_LT(fabsf(cost_diff), 1e-3 * trajectory_costs_cpu(i))
                << error_prefix << ": CPU = " << trajectory_costs_cpu(i) << ", GPU = " << trajectory_costs_gpu(i)
                << std::endl;
          }
        }
      }
    }
  }
}

// TEST_F(RMPPIKernels, InitEvalRollout)
// {
//   // Given the initial states, we need to roll out the number of samples.
//   // 1.)  Generate the noise used to evaluate each sample.
//   const int num_candidates = 9;

//   float dt = 0.01;
//   float lambda = 0.75;
//   float alpha = 0.5;
//   int crash_status[1] = { 0 };

//   Eigen::Matrix<float, 4, num_candidates> x0_candidates;
//   x0_candidates << -4, -3, -2, -1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0,
//       0, 0, 0, 0;

//   // For each candidate, we want to estimate the free energy using a set number of samples.
//   const int num_samples = 64;

//   // We are going to propagate a trajectory for a given number of timesteps
//   const int num_timesteps = 5;

//   // Call the GPU setup functions of the model and cost
//   model->GPUSetup();
//   cost->GPUSetup();
//   sampler->GPUSetup();

//   // Allocate and deallocate the CUDA memory
//   int* strides_d;
//   float* exploration_var_d;
//   float* states_d;
//   float* control_d;
//   float* control_noise_d;
//   float* costs_d;

//   HANDLE_ERROR(cudaMalloc((void**)&strides_d, sizeof(int) * num_candidates));
//   HANDLE_ERROR(cudaMalloc((void**)&exploration_var_d, sizeof(float) * dynamics::CONTROL_DIM));
//   HANDLE_ERROR(cudaMalloc((void**)&states_d, sizeof(float) * dynamics::STATE_DIM * num_candidates));
//   HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float) * dynamics::CONTROL_DIM * num_timesteps));
//   HANDLE_ERROR(cudaMalloc((void**)&control_noise_d,
//                           sizeof(float) * dynamics::CONTROL_DIM * num_candidates * num_timesteps * num_samples));
//   HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_samples * num_candidates));

//   // We need to generate a nominal trajectory for the control
//   Eigen::Matrix<float, dynamics::CONTROL_DIM, num_timesteps> nominal_control =
//       Eigen::MatrixXf::Random(dynamics::CONTROL_DIM, num_timesteps);

//   // std::cout << "Nominal Control" << nominal_control << std::endl;

//   // Exploration variance
//   Eigen::Matrix<float, dynamics::CONTROL_DIM, 1> exploration_var;
//   exploration_var << 2, 2;
//   auto sampler_params = sampler->getParams();
//   for (int i = 0; i < dynamics::CONTROL_DIM; i++)
//   {
//     sampler_params.std_dev[i] = exploration_var[i];
//   }
//   sampler->setParams(sampler_params);

//   // std::cout << exploration_var << std::endl;

//   // Generate noise to perturb the nominal control
//   // Seed the PseudoRandomGenerator with the CPU time.
//   curandGenerator_t gen_;
//   curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
//   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//   curandSetPseudoRandomGeneratorSeed(gen_, seed);
//   curandGenerateNormal(gen_, control_noise_d, num_samples * num_candidates * num_timesteps * dynamics::CONTROL_DIM,
//   0.0,
//                        1.0);

//   // Copy the noise back to the CPU so we can use it!
//   Eigen::Matrix<float, dynamics::CONTROL_DIM, num_samples * num_candidates * num_timesteps> control_noise;

//   std::vector<float> control_noise_data(num_samples * num_candidates * num_timesteps * dynamics::CONTROL_DIM);

//   HANDLE_ERROR(cudaMemcpy(control_noise_data.data(), control_noise_d,
//                           sizeof(float) * num_candidates * num_samples * num_timesteps * dynamics::CONTROL_DIM,
//                           cudaMemcpyDeviceToHost));

//   control_noise = Eigen::Map<Eigen::Matrix<float, dynamics::CONTROL_DIM, num_candidates * num_samples *
//   num_timesteps>>(
//       control_noise_data.data());

//   int ctrl_stride = 2;

//   Eigen::Matrix<int, 1, 9> strides;
//   strides << 1, 2, 3, 4, 4, 4, 4, 4, 4;

//   // Let us make temporary variables to hold the states and state derivatives and controls
//   dynamics::state_array x_current, x_dot_current, x_next;
//   dynamics::output_array output = dynamics::output_array::Zero();
//   dynamics::control_array u_current;
//   dynamics::control_array noise_current;

//   Eigen::Matrix<float, 1, num_samples * 9> cost_vector;

//   float cost_current = 0.0;
//   for (int i = 0; i < 9; ++i)
//   {  // Iterate through each candidate
//     Eigen::Matrix<float, dynamics::CONTROL_DIM, num_timesteps> candidate_nominal_control;
//     // For each candidate we want to slide the controls according to their own stride.
//     for (int k = 0; k < num_timesteps; ++k)
//     {
//       if (k + strides(i) >= num_timesteps)
//       {
//         candidate_nominal_control.col(k) = nominal_control.col(num_timesteps - 1);
//       }
//       else
//       {
//         candidate_nominal_control.col(k) = nominal_control.col(k + strides(i));
//       }
//     }

//     for (int j = 0; j < num_samples; ++j)
//     {
//       x_current = x0_candidates.col(i);  // The initial state of the rollout
//       for (int k = 0; k < num_timesteps; ++k)
//       {
//         // get the control plus a disturbance
//         if (j == 0 || k < ctrl_stride)
//         {  // First sample should always be noise free as should any timesteps that are below the control stride
//           noise_current = dynamics::control_array::Zero();
//         }
//         else
//         {
//           noise_current = control_noise.col((i * num_samples + j) * num_timesteps + k).cwiseProduct(exploration_var);
//         }
//         u_current = candidate_nominal_control.col(k) + noise_current;

//         // enforce constraints
//         model->enforceConstraints(x_current, u_current);

//         // compute the cost
//         if (k > 0)
//         {
//           cost_current +=
//               (cost->computeRunningCost(output, candidate_nominal_control.col(k), k, crash_status) - cost_current) /
//               (1.0f * k);
//         }
//         // Update State
//         model->step(x_current, x_next, x_dot_current, u_current, output, k, dt);
//         x_current = x_next;
//       }
//       // compute the terminal cost -> this is the free energy estimate, save it!
//       cost_vector.col(i * num_samples + j) << cost_current + cost->terminalCost(output) / (num_timesteps - 1);
//       cost_current = 0.0;
//     }
//   }

//   // Copy the state candidates to GPU
//   HANDLE_ERROR(cudaMemcpy(states_d, x0_candidates.data(), sizeof(float) * dynamics::STATE_DIM * num_candidates,
//                           cudaMemcpyHostToDevice));

//   // Copy the control to the GPU
//   HANDLE_ERROR(cudaMemcpy(control_d, nominal_control.data(), sizeof(float) * dynamics::CONTROL_DIM * num_timesteps,
//                           cudaMemcpyHostToDevice));

//   // Copy the strides to the GPU
//   HANDLE_ERROR(cudaMemcpy(strides_d, strides.data(), sizeof(int) * num_candidates, cudaMemcpyHostToDevice));

//   // Copy exploration variance to GPU
//   HANDLE_ERROR(cudaMemcpy(exploration_var_d, exploration_var.data(), sizeof(float) * dynamics::CONTROL_DIM,
//                           cudaMemcpyHostToDevice));

//   // Run the GPU test kernel of the init eval kernel and get the output data
//   // ();
//   mppi::kernels::rmppi::launchInitEvalKernel<dynamics, cost_function, SAMPLER>(
//       model->model_d_, cost->cost_d_, sampler->sampling_d_, dt, num_timesteps, num_samples * num_candidates,
//       lambda, alpha, num_samples, strides_d, states_d, costs_d, dim3(64, 4, 1), 0, true);
//   // rmppi_kernels::launchInitEvalKernel<dynamics, cost_function, 64, 8, num_samples>(
//   //     model->model_d_, cost->cost_d_, num_candidates, num_timesteps, lambda, alpha, ctrl_stride, dt, strides_d,
//   //     exploration_var_d, states_d, control_d, control_noise_d, costs_d, 0);

//   CudaCheckError();

//   Eigen::Matrix<float, 1, num_samples * num_candidates> cost_vector_GPU;
//   // Compare with the CPU version
//   HANDLE_ERROR(cudaMemcpy(cost_vector_GPU.data(), costs_d, sizeof(float) * num_samples * num_candidates,
//                           cudaMemcpyDeviceToHost));
//   auto cost_diff = cost_vector - cost_vector_GPU;
//   EXPECT_LT(cost_diff.norm(), 5e-3);
// }

// TEST(RMPPITest, RMPPIRolloutKernel_CPU_v_GPU)
// {
//   using DYN = DoubleIntegratorDynamics;
//   using COST = DoubleIntegratorCircleCost;
//   using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
//   DYN model;
//   COST cost;
//   SAMPLER sampler;

//   const int num_timesteps = 50;
//   float dt = 0.01;
//   using FB_T = DDPFeedback<DYN, num_timesteps>;
//   FB_T fb_controller(&model, dt);

//   const int state_dim = DYN::STATE_DIM;
//   const int control_dim = DYN::CONTROL_DIM;

//   // int max_iter = 10;
//   float lambda = 1.0;
//   float alpha = 0.1;
//   const int num_rollouts = 64;
//   int optimization_stride = 1;

//   float sigma_u[control_dim] = { 0.5, 0.05 };  // variance to sample noise from
//   // float fb_u[num_rollouts * control_dim * state_dim];

//   DYN::state_array x_init_act;
//   x_init_act << 2, 0, 0, 0;
//   DYN::state_array x_init_nom;
//   x_init_nom << 2, 0, 0.1, 0;

//   // Generate control noise
//   float sampled_noise[num_rollouts * num_timesteps * control_dim];
//   std::mt19937 rng_gen;
//   std::vector<std::normal_distribution<float>> control_dist;
//   for (int i = 0; i < control_dim; i++)
//   {
//     control_dist.push_back(std::normal_distribution<float>(0, 1));
//   }

//   for (int n = 0; n < num_rollouts; n++)
//   {
//     int n_ind = n * num_timesteps * control_dim;
//     for (int t = 0; t < num_timesteps; t++)
//     {
//       int t_ind = t * control_dim;
//       for (int j = 0; j < control_dim; j++)
//       {
//         sampled_noise[n_ind + t_ind + j] = control_dist[j](rng_gen);
//       }
//     }
//   }
//   // TODO: Figure out nonzero Initial control trajectory
//   float u_traj[num_timesteps * control_dim] = { 0 };

//   for (int i = 0; i < num_timesteps * control_dim; i++)
//   {
//     u_traj[i] = 2.0;
//   }

//   u_traj[0] = 1;
//   u_traj[1] = 0.5;

//   u_traj[10] = 1;
//   u_traj[11] = 0.5;

//   u_traj[14] = -1;
//   u_traj[15] = 0.5;

//   auto fb_state = fb_controller.getFeedbackStatePointer();
//   for (int i = 0; i < num_timesteps * state_dim * control_dim; i++)
//   {
//     fb_state->fb_gain_traj_[i] = -15;
//   }

//   /**
//    * Create vectors of data for GPU/CPU test
//    */
//   std::vector<float> x_init_act_vec, x_init_nom_vec, sigma_u_vec, u_traj_vec;
//   x_init_act_vec.assign(x_init_act.data(), x_init_act.data() + state_dim);
//   x_init_nom_vec.assign(x_init_nom.data(), x_init_nom.data() + state_dim);
//   sigma_u_vec.assign(sigma_u, sigma_u + control_dim);
//   u_traj_vec.assign(u_traj, u_traj + num_timesteps * control_dim);
//   std::vector<float> feedback_gains_seq_vec, sampled_noise_vec;
//   int control_traj_size = num_rollouts * num_timesteps * control_dim;

//   sampled_noise_vec.reserve(control_traj_size * 2);
//   for (int i = 0; i < control_traj_size; i++)
//   {
//     sampled_noise_vec[i] = sampled_noise[i];
//     sampled_noise_vec[control_traj_size + i] = sampled_noise_vec[i];
//   }

//   float value_func_threshold = 50000;

//   //  std::cout <<  "X_init_act_vec " << std::endl;
//   //  for (int i = 0; i < x_init_act_vec.size(); ++i) {
//   //    std::cout <<  " " << x_init_act_vec[i];
//   //  }
//   //  std::cout << std::endl;
//   //
//   //  std::cout <<  "X_init_nom_vec " << std::endl;
//   //  for (int i = 0; i < x_init_nom_vec.size(); ++i) {
//   //    std::cout <<  " " << x_init_nom_vec[i];
//   //  }
//   //  std::cout << std::endl;

//   // Output Trajectory Costs
//   std::array<float, num_rollouts> costs_act_GPU, costs_nom_GPU;
//   std::array<float, num_rollouts> costs_act_CPU, costs_nom_CPU;
//   launchRMPPIRolloutKernelGPU<DYN, COST, SAMPLER, FB_T, num_rollouts>(
//       &model, &cost, &sampler, &fb_controller, dt, num_timesteps, optimization_stride, lambda, alpha,
//       value_func_threshold, x_init_act_vec, x_init_nom_vec, sigma_u_vec, u_traj_vec, sampled_noise_vec,
//       costs_act_GPU, costs_nom_GPU);
//   launchRMPPIRolloutKernelCPU<DYN, COST, FB_T, num_rollouts>(
//       &model, &cost, &fb_controller, dt, num_timesteps, optimization_stride, lambda, alpha, value_func_threshold,
//       x_init_act_vec, x_init_nom_vec, sigma_u_vec, u_traj_vec, sampled_noise_vec, costs_act_CPU, costs_nom_CPU);

//   //  for (int i = 0; i < costs_nom_CPU.size(); ++i) {
//   //    std::cout << "Nominal Cost CPU: " << costs_nom_CPU[i] << std::endl;
//   //    std::cout << "Nominal Cost GPU: " << costs_nom_GPU[i] << std::endl;
//   //  }

//   float max_diff_nom = -100;
//   float max_diff_act = -100;
//   int diff_nom_ind = -1;
//   int diff_act_ind = -1;
//   for (int i = 0; i < num_rollouts; i++)
//   {
//     // std::cout << i << ": GPU Nom: " << costs_nom_GPU[i] << ", CPU Nom: " << costs_nom_CPU[i] << std::endl;
//     float diff_nom = std::abs(costs_nom_CPU[i] - costs_nom_GPU[i]);
//     float diff_act = std::abs(costs_act_CPU[i] - costs_act_GPU[i]);
//     if (diff_nom > max_diff_nom)
//     {
//       max_diff_nom = diff_nom;
//       diff_nom_ind = i;
//     }
//     if (diff_act > max_diff_act)
//     {
//       max_diff_act = diff_act;
//       diff_act_ind = i;
//     }
//   }
//   std::cout << "Max Real Difference between CPU and GPU rollout " << diff_act_ind << ": " << max_diff_act <<
//   std::endl; std::cout << "Max Nominal Difference between CPU and GPU rollout " << diff_nom_ind << ": " <<
//   max_diff_nom
//             << std::endl;
//   array_assert_float_eq<num_rollouts>(costs_act_GPU, costs_act_CPU);
//   std::cout << "Checking nominal systems differences between CPU and GPU" << std::endl;
//   array_assert_float_eq<num_rollouts>(costs_nom_GPU, costs_nom_CPU);
// }

// TEST(RMPPITest, TwoSystemRolloutKernelComparison)
// {
//   /**
//    * If the nominal state and the real state are equal, and we are using the
//    * same noise between the two, then the output result should be equal to the
//    * standard rollout kernel.
//    */
//   using DYN = DoubleIntegratorDynamics;
//   using COST = DoubleIntegratorCircleCost;
//   const int num_timesteps = 100;
//   using FB_T = DDPFeedback<DYN, num_timesteps>;

//   DYN model;
//   COST cost;
//   float dt = 0.01;
//   FB_T fb_controller(&model, dt);

//   model.GPUSetup();
//   cost.GPUSetup();
//   fb_controller.GPUSetup();

//   const int state_dim = DYN::STATE_DIM;
//   const int control_dim = DYN::CONTROL_DIM;

//   float lambda = 4.2;
//   float alpha = 0.05;
//   const int num_rollouts = 256;
//   int optimization_stride = 1;

//   std::array<float, control_dim> sigma_u = { 0.5, 1.5 };

//   std::array<float, state_dim> x_real = { 2, 0, 1, 1 };
//   std::array<float, state_dim> x_nominal = { 2, 0, 1, 1 };

//   std::array<float, control_dim * num_timesteps> u_init_trajectory{};
//   std::default_random_engine generator(7.0);
//   std::normal_distribution<float> distribution(0.0, 1.0);
//   for (auto& u_init : u_init_trajectory)
//   {
//     u_init = 2 * distribution(generator);
//   }

//   std::array<float, num_timesteps * num_rollouts * control_dim> control_noise_array{};
//   for (auto& noise : control_noise_array)
//   {
//     noise = distribution(generator);
//   }

//   // Create some random feedback gains
//   using feedback_gain_traj_matrix = Eigen::Matrix<float, control_dim * state_dim * num_timesteps, 1>;
//   feedback_gain_traj_matrix random_noise = feedback_gain_traj_matrix::Random();
//   auto fb_state = fb_controller.getFeedbackStatePointer();
//   for (int i = 0; i < num_timesteps * state_dim * control_dim; i++)
//   {
//     fb_state->fb_gain_traj_[i] = random_noise.data()[i];
//   }
//   fb_controller.copyToDevice();

//   // Create objects that will hold the results
//   std::array<float, 2 * num_rollouts> rmppi_costs_out{};
//   std::array<float, num_rollouts> mppi_costs_out{};

//   // Launch the test kernel...
//   launchComparisonRolloutKernelTest<DYN, COST, FB_T, num_rollouts, num_timesteps, 64, 8>(
//       &model, &cost, &fb_controller, dt, lambda, alpha, x_real, x_nominal, u_init_trajectory, control_noise_array,
//       sigma_u, rmppi_costs_out, mppi_costs_out, optimization_stride, 0);

//   for (int i = 0; i < num_rollouts; i++)
//   {
//     ASSERT_FLOAT_EQ(rmppi_costs_out[num_rollouts + i], rmppi_costs_out[i]) << i;
//   }

//   for (int i = 0; i < num_rollouts; i++)
//   {
//     EXPECT_NEAR(rmppi_costs_out[i], mppi_costs_out[i], 1e-1) << i;
//   }

//   for (int i = 0; i < num_rollouts; i++)
//   {
//     EXPECT_NEAR(rmppi_costs_out[num_rollouts + i], mppi_costs_out[i], 1e-1) << num_rollouts + i;
//   }
// }
