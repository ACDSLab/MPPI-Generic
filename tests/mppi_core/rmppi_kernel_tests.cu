//
// Created by mgandhi on 5/23/20.
//
#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/core/rmppi_kernel_test.cuh>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/utils/test_helper.h>

class RMPPIKernels : public ::testing::Test {
public:
  using dynamics = DoubleIntegratorDynamics;
  using cost_function = DoubleIntegratorCircleCost;

  void SetUp() override {
    model = new dynamics(10);  // Initialize the double integrator dynamics
    cost = new cost_function;  // Initialize the cost function
  }

  void TearDown() override {
    delete model;
    delete cost;
  }

  dynamics* model;
  cost_function* cost;
};

TEST_F(RMPPIKernels, InitEvalRollout) {
  // Given the initial states, we need to roll out the number of samples.
  // 1.)  Generate the noise used to evaluate each sample.
  //

  const int num_candidates = 9;

  float dt = 0.01;

  Eigen::Matrix<float, 4, num_candidates> x0_candidates;
  x0_candidates << -4 , -3, -2, -1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4, 4, 4,
  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // For each candidate, we want to estimate the free energy using a set number of samples.
  const int num_samples = 64;

  // We are going to propagate a trajectory for a given number of timesteps
  const int num_timesteps = 5;

  // Call the GPU setup functions of the model and cost
  model->GPUSetup();
  cost->GPUSetup();

  // Allocate and deallocate the CUDA memory
  int* strides_d;
  float* exploration_var_d;
  float* states_d;
  float* control_d;
  float* control_noise_d;
  float* costs_d;
  HANDLE_ERROR(cudaMalloc((void**)&strides_d, sizeof(int)*num_candidates));
  HANDLE_ERROR(cudaMalloc((void**)&exploration_var_d, sizeof(float)*dynamics::CONTROL_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&states_d, sizeof(float)*dynamics::STATE_DIM*num_candidates));
  HANDLE_ERROR(cudaMalloc((void**)&control_d, sizeof(float)*dynamics::CONTROL_DIM*num_timesteps));
  HANDLE_ERROR(cudaMalloc((void**)&control_noise_d, sizeof(float)*dynamics::CONTROL_DIM*num_candidates*num_timesteps*num_samples));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float)*num_samples*num_candidates));

  // We need to generate a nominal trajectory for the control
  Eigen::Matrix<float, dynamics::CONTROL_DIM, num_timesteps> nominal_control = Eigen::MatrixXf::Random(dynamics::CONTROL_DIM, num_timesteps);

//  std::cout << "Nominal Control" << nominal_control << std::endl;

  // Exploration variance
  Eigen::Matrix<float, dynamics::CONTROL_DIM, 1> exploration_var;
  exploration_var << 2, 2;

//  std::cout << exploration_var << std::endl;

  // Generate noise to perturb the nominal control
  // Seed the PseudoRandomGenerator with the CPU time.
  curandGenerator_t gen_;
  curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  curandSetPseudoRandomGeneratorSeed(gen_, seed);
  curandGenerateNormal(gen_, control_noise_d,
                       num_samples*num_candidates*num_timesteps*dynamics::CONTROL_DIM,
                       0.0, 1.0);

  // Copy the noise back to the CPU so we can use it!
  Eigen::Matrix<float, dynamics::CONTROL_DIM, num_samples*num_candidates*num_timesteps> control_noise;

  std::vector<float> control_noise_data(num_samples*num_candidates*num_timesteps*dynamics::CONTROL_DIM);

  HANDLE_ERROR(cudaMemcpy(control_noise_data.data(), control_noise_d, sizeof(float)*num_candidates*num_samples*num_timesteps*dynamics::CONTROL_DIM, cudaMemcpyDeviceToHost));

  control_noise =  Eigen::Map<Eigen::Matrix<float,dynamics::CONTROL_DIM, num_candidates*num_samples*num_timesteps>>(control_noise_data.data());

//   std::cout << "Control Noise\n" << control_noise.col(num_samples*num_timesteps).cwiseProduct(exploration_var).transpose() << std::endl;
  int ctrl_stride = 2;

  Eigen::Matrix<int, 1, 9> strides;
  strides << 1, 2, 3, 4, 4, 4, 4, 4, 4;


  // Let us make temporary variables to hold the states and state derivatives and controls
  dynamics::state_array x_current, x_dot_current;
  dynamics::control_array u_current;

  Eigen::Matrix<float, 1, num_samples*9> cost_vector;

  float cost_current = 0.0;
  for (int i = 0; i < 9; ++i) { // Iterate through each candidate
    Eigen::Matrix<float, dynamics::CONTROL_DIM, num_timesteps> candidate_nominal_control;
    // For each candidate we want to slide the controls according to their own stride.
    for (int k = 0; k < num_timesteps; ++k) {
      if (k + strides(i) >= num_timesteps) {
        candidate_nominal_control.col(k) = nominal_control.col(num_timesteps-1);
      } else {
        candidate_nominal_control.col(k) = nominal_control.col(k+strides(i));
      }
    }
//    if (i == 0) {
//      std::cout << "Nominal_control:\n" << nominal_control << std::endl;
//      std::cout << "Candidate_control:\n" << candidate_nominal_control << std::endl;
//    }
    for (int j = 0; j < num_samples; ++j) {
      x_current = x0_candidates.col(i);  // The initial state of the rollout
      for (int k = 0; k < num_timesteps; ++k) {
        // compute the cost
        if (k > 0) {
          cost_current += (cost->computeStateCost(x_current) * dt - cost_current) / (1.0*k);
        }
        // get the control plus a disturbance
        if (j == 0 || k < ctrl_stride) { // First sample should always be noise free as should any timesteps that are below the control stride
          u_current = candidate_nominal_control.col(k);
        } else {
          u_current = candidate_nominal_control.col(k) +
                      control_noise.col(i * num_samples * num_timesteps + j * num_timesteps + k).cwiseProduct(
                              exploration_var);
        }

        // compute the next state_dot
        model->computeDynamics(x_current, u_current, x_dot_current);
        // update the state to the next
        model->updateState(x_current, x_dot_current, dt);
        }
      // compute the terminal cost -> this is the free energy estimate, save it!
      cost_vector.col(i*num_samples + j) << cost_current;
      cost_current = 0.0;
    }
  }


//  std::cout << "Eigen strides: " << strides << std::endl;


  // Copy the state candidates to GPU
  HANDLE_ERROR(cudaMemcpy(states_d, x0_candidates.data(), sizeof(float)*dynamics::STATE_DIM*num_candidates, cudaMemcpyHostToDevice));

  // Copy the control to the GPU
  HANDLE_ERROR(cudaMemcpy(control_d, nominal_control.data(), sizeof(float)*dynamics::CONTROL_DIM*num_timesteps, cudaMemcpyHostToDevice));

  // Copy the strides to the GPU
  HANDLE_ERROR(cudaMemcpy(strides_d, strides.data(), sizeof(int)*num_candidates, cudaMemcpyHostToDevice));

  // Copy exploration variance to GPU
  HANDLE_ERROR(cudaMemcpy(exploration_var_d, exploration_var.data(), sizeof(float)*dynamics::CONTROL_DIM, cudaMemcpyHostToDevice));

  // Run the GPU test kernel of the init eval kernel and get the output data
  // ();
  rmppi_kernels::launchInitEvalKernel<dynamics, cost_function, 64, 8, num_samples>(model->model_d_, cost->cost_d_,
          num_candidates, num_timesteps, ctrl_stride, dt,
          strides_d, exploration_var_d, states_d, control_d, control_noise_d, costs_d);
  CudaCheckError();

  Eigen::Matrix<float, 1, num_samples*num_candidates> cost_vector_GPU;
  // Compare with the CPU version
  HANDLE_ERROR(cudaMemcpy(cost_vector_GPU.data(), costs_d, sizeof(float)*num_samples*num_candidates, cudaMemcpyDeviceToHost));

//  std::cout <<  "Cost Vector CPU\n" << cost_vector.col(65) << std::endl;
//  std::cout << "Cost Vector GPU\n" << cost_vector_GPU.col(65) << std::endl;

//  std::cout << (cost_vector - cost_vector_GPU).transpose() << std::endl;

  EXPECT_LT((cost_vector - cost_vector_GPU).norm(), 1e-4);
}

TEST(RMPPITest, CPURolloutKernel) {
  using DYN = DoubleIntegratorDynamics;
  using COST = DoubleIntegratorCircleCost;
  DYN model;
  COST cost;

  const int state_dim = DYN::STATE_DIM;
  const int control_dim = DYN::CONTROL_DIM;

  float dt = 0.01;
  // int max_iter = 10;
  float lambda = 0.1;
  const int num_timesteps = 50;
  const int num_rollouts = 64;

  // float x[num_rollouts * state_dim * 2];
  // float x_dot[num_rollouts * state_dim * 2];
  // float u[num_rollouts * control_dim * 2];
  // float du[num_rollouts * control_dim * 2];
  float sigma_u[control_dim] = {0.5, 0.05}; // variance to sample noise from
  COST::control_matrix cost_variance = COST::control_matrix::Identity();
  for(int i = 0; i < control_dim; i++) {
    cost_variance(i, i) = sigma_u[i];
  }
  // float fb_u[num_rollouts * control_dim * state_dim];

  DYN::state_array x_init_act;
  x_init_act << 0, 0, 0, 0;
  DYN::state_array x_init_nom;
  x_init_nom << 0, 0, 0.1, 0;

  // Generate control noise
  float sampled_noise[num_rollouts * num_timesteps * control_dim];
  std::mt19937 rng_gen;
  std::vector<std::normal_distribution<float>> control_dist;
  for (int i = 0; i < control_dim; i++) {
    control_dist.push_back(std::normal_distribution<float>(0, 1));
  }

  for (int n = 0; n < num_rollouts; n++) {
    int n_ind = n * num_timesteps * control_dim;
    for (int t = 0; t < num_timesteps; t++) {
      int t_ind = t * control_dim;
      for (int j = 0; j < control_dim; j++) {
        sampled_noise[n_ind + t_ind + j] = control_dist[j](rng_gen);
      }
    }
  }
  // TODO: Figure out nonzero Initial control trajectory
  float u_traj[num_timesteps * control_dim] = {0};
  u_traj[0] = 1;
  u_traj[1] = 0.5;

  u_traj[10] = 1;
  u_traj[11] = 0.5;

  u_traj[14] = -1;
  u_traj[15] = 0.5;

  // TODO: Generate feedback gain trajectories
  VanillaMPPIController<DYN, COST, 100, 512, 64, 8>::K_matrix feedback_gains;
  for (int i = 0; i < num_timesteps; i++) {
    feedback_gains.push_back(DYN::feedback_matrix::Constant(-15));
  }

  // Copy Feedback Gains into an array
  float feedback_array[num_timesteps * control_dim * state_dim];
  for (size_t i = 0; i < feedback_gains.size(); i++) {
    // std::cout << "Matrix " << i << ":\n";
    // std::cout << feedback_gains[i] << std::endl;
    int i_index = i * control_dim * state_dim;

    for (size_t j = 0; j < control_dim * state_dim; j++) {
      feedback_array[i_index + j] = feedback_gains[i].data()[j];
    }
  }
  /**
   * Create vectors of data for GPU/CPU test
   */
  std::vector<float> x_init_act_vec, x_init_nom_vec, sigma_u_vec, u_traj_vec;
  x_init_act_vec.assign(x_init_act.data(), x_init_act.data() + state_dim);
  x_init_nom_vec.assign(x_init_nom.data(), x_init_nom.data() + state_dim);
  sigma_u_vec.assign(sigma_u, sigma_u + control_dim);
  u_traj_vec.assign(u_traj, u_traj + num_timesteps * control_dim);
  std::vector<float> feedback_gains_seq_vec, sampled_noise_vec;
  feedback_gains_seq_vec.assign(feedback_array, feedback_array +
    num_timesteps * control_dim * state_dim);
  sampled_noise_vec.assign(sampled_noise, sampled_noise +
    num_rollouts * num_timesteps * control_dim);

  float value_func_threshold = 50000;

  // Output Trajectory Costs
  std::array<float, num_rollouts> costs_act_GPU, costs_nom_GPU;
  std::array<float, num_rollouts> costs_act_CPU, costs_nom_CPU;
  launchRMPPIRolloutKernelGPU<DYN, COST, num_rollouts>(&model, &cost, dt,
    num_timesteps, lambda, value_func_threshold, x_init_act_vec, x_init_nom_vec,
    sigma_u_vec, u_traj_vec, feedback_gains_seq_vec, sampled_noise_vec,
    costs_act_GPU, costs_nom_GPU);
  launchRMPPIRolloutKernelCPU<DYN, COST, num_rollouts>(&model, &cost, dt,
    num_timesteps, lambda, value_func_threshold, x_init_act_vec, x_init_nom_vec,
    sigma_u_vec, u_traj_vec, feedback_gains_seq_vec, sampled_noise_vec,
    costs_act_CPU, costs_nom_CPU);

  float max_diff_nom = -100;
  float max_diff_act = -100;
  int diff_nom_ind = -1;
  int diff_act_ind = -1;
  for (int i = 0; i < num_rollouts; i++) {
    // std::cout << i << ": GPU Nom: " << costs_nom_GPU[i] << ", CPU Nom: " << costs_nom_CPU[i] << std::endl;
    float diff_nom  = std::abs(costs_nom_CPU[i] - costs_nom_GPU[i]);
    float diff_act  = std::abs(costs_act_CPU[i] - costs_act_GPU[i]);
    if (diff_nom > max_diff_nom) {
      max_diff_nom = diff_nom;
      diff_nom_ind = i;
    }
    if (diff_act > max_diff_act) {
      max_diff_act = diff_act;
      diff_act_ind = i;
    }
  }
  std::cout << "Max Real Difference between CPU and GPU rollout "<< diff_act_ind
            << ": " << max_diff_act << std::endl;
  std::cout << "Max Nominal Difference between CPU and GPU rollout "<< diff_nom_ind
            << ": " << max_diff_nom << std::endl;
  array_assert_float_eq<num_rollouts>(costs_act_GPU, costs_act_CPU);
  std::cout << "Checking nominal systems differences between CPU and GPU" << std::endl;
  array_assert_float_eq<num_rollouts>(costs_nom_GPU, costs_nom_CPU);
}