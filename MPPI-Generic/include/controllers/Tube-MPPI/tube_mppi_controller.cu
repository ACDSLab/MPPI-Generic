
#include <mppi_core/mppi_common.cuh>

#define TubeMPPI TubeMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>
#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
        int BDIM_X, int BDIM_Y>
TubeMPPI::TubeMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                             float gamma, int num_timesteps,
                             const StateCostWeight& Q,
                             const Hessian& Qf,
                             const ControlCostWeight& R,
                             const control_array& control_variance,
                             const control_trajectory& init_control_traj,
                             cudaStream_t stream) :
dt_(dt), num_iters_(max_iter), gamma_(gamma),
actual_control_(init_control_traj),
nominal_control_(init_control_traj), stream_(stream) {
    this->model_ = model;
    this->cost_ = cost;

    this->control_variance_ = control_variance;
    this->num_timesteps_ = num_timesteps;

    this->setNumTimesteps(num_timesteps);

    // Create the random number generator
    this->createAndSeedCUDARandomNumberGen();

    // Bind the model and control to the given stream
    this->setCUDAStream(stream);

    // Call the GPU setup functions of the model and cost
    this->model_->GPUSetup();
    this->cost_->GPUSetup();


    // Allocate CUDA memory for the controller
    allocateCUDAMemory();

    // Copy the noise variance to the device
    this->copyControlVarianceToDevice();

    // TODO: CREATE DDP STUFF HERE
    initDDP(Q, Qf, R);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::computeControl(const state_array& state) {
    if (!nominalStateInit_){
      for (int i = 0; i < DYN_T::STATE_DIM; i++){
        nominal_state_(i, 0) = state(i);
      }
      nominalStateInit_ = true;
    }

    // Handy reference pointers
    float * trajectory_costs_nominal_d = trajectory_costs_d_ + NUM_ROLLOUTS;
    float * initial_state_nominal_d = initial_state_d_ + DYN_T::STATE_DIM;

    float * control_noise_nominal_d = control_noise_d_ + NUM_ROLLOUTS *
                                      this->num_timesteps_ * DYN_T::CONTROL_DIM;
    float * control_nominal_d = control_d_ + this->num_timesteps_ * DYN_T::CONTROL_DIM;

    // Send the initial condition to the device

    HANDLE_ERROR( cudaMemcpyAsync(initial_state_d_, state.data(),
        DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));

    HANDLE_ERROR( cudaMemcpyAsync(initial_state_nominal_d, nominal_state_.data(),
        DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));

    for (int opt_iter = 0; opt_iter < num_iters_; opt_iter++) {
        // Send the nominal control to the device
        copyControlToDevice();

        //Generate noise data
        curandGenerateNormal(this->gen_, control_noise_d_,
                             NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM,
                             0.0, 1.0);
        HANDLE_ERROR( cudaMemcpyAsync(control_noise_nominal_d, control_noise_d_,
                     NUM_ROLLOUTS*this->num_timesteps_*DYN_T::CONTROL_DIM * sizeof(float),
                     cudaMemcpyDeviceToDevice,
                     stream_) );

        //Launch the rollout kernel
        mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y, 2>(
            this->model_->model_d_, this->cost_->cost_d_, dt_, this->num_timesteps_,
            initial_state_d_, control_d_, control_noise_d_,
            this->control_variance_d_, trajectory_costs_d_, stream_);

        // Copy the costs back to the host
        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_actual_.data(),
            trajectory_costs_d_,
            NUM_ROLLOUTS*sizeof(float),
            cudaMemcpyDeviceToHost, stream_));

        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(),
            trajectory_costs_nominal_d,
            NUM_ROLLOUTS*sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR( cudaStreamSynchronize(stream_) );

        baseline_actual_ = mppi_common::computeBaselineCost(
            trajectory_costs_actual_.data(),
            NUM_ROLLOUTS);

        baseline_nominal_ = mppi_common::computeBaselineCost(
            trajectory_costs_nominal_.data(),
            NUM_ROLLOUTS);

        // Launch the norm exponential kernel for both actual and nominal
        mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
            trajectory_costs_d_, gamma_, baseline_actual_, stream_);

        mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
            trajectory_costs_nominal_d, gamma_, baseline_nominal_, stream_);

        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_actual_.data(),
            trajectory_costs_d_,
            NUM_ROLLOUTS*sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(),
            trajectory_costs_nominal_d,
            NUM_ROLLOUTS*sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR(cudaStreamSynchronize(stream_));

        // Compute the normalizer
        normalizer_actual_ = mppi_common::computeNormalizer(
            trajectory_costs_actual_.data(), NUM_ROLLOUTS);
        normalizer_nominal_ = mppi_common::computeNormalizer(
            trajectory_costs_nominal_.data(), NUM_ROLLOUTS);

        // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
        mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
            trajectory_costs_d_, control_noise_d_, control_d_,
            normalizer_actual_, this->num_timesteps_, stream_);
        mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
            trajectory_costs_nominal_d,
            control_noise_nominal_d, control_nominal_d,
            normalizer_nominal_, this->num_timesteps_, stream_);

        // Transfer the new control to the host
        HANDLE_ERROR( cudaMemcpyAsync(actual_control_.data(), control_d_,
                sizeof(float)*this->num_timesteps_*DYN_T::CONTROL_DIM,
                cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR( cudaMemcpyAsync(nominal_control_.data(), control_nominal_d,
                sizeof(float)*this->num_timesteps_*DYN_T::CONTROL_DIM,
                cudaMemcpyDeviceToHost, stream_));
        cudaStreamSynchronize(stream_);

        if (baseline_actual_ < baseline_nominal_ + nominal_threshold_) {
            use_nominal_state_ = false;
            // reset nominal to actual

            // std::copy(actual_state_.begin(), actual_state_.end(), nominal_state_.begin());
            // std::copy(actual_control_.begin(), actual_control_.end(), nominal_control_.begin());
            nominal_state_ = actual_state_;
            nominal_control_ = actual_control_;
        }



        // TODO Add SavitskyGolay?

        // TODO Add nominal state computation

    }

}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::copyControlToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(control_d_, actual_control_.data(),
                                 sizeof(float)*actual_control_.size(),
                                 cudaMemcpyHostToDevice, stream_));

    HANDLE_ERROR(cudaMemcpyAsync(control_d_ + nominal_control_.size(),
                                 nominal_control_.data(),
                                 sizeof(float)*nominal_control_.size(),
                                 cudaMemcpyHostToDevice, stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::allocateCUDAMemory() {
    HANDLE_ERROR(cudaMalloc((void**)&initial_state_d_,
                            sizeof(float) * DYN_T::STATE_DIM * 2));
    HANDLE_ERROR(cudaMalloc((void**)&control_d_,
                            sizeof(float) * DYN_T::CONTROL_DIM *
                            this->num_timesteps_ * 2));
    HANDLE_ERROR(cudaMalloc((void**)&state_d_,
                            sizeof(float) * DYN_T::STATE_DIM *
                            this->num_timesteps_ * 2));
    HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d_,
                            sizeof(float) * NUM_ROLLOUTS * 2));
    HANDLE_ERROR(cudaMalloc((void**)&this->control_variance_d_,
                            sizeof(float) * DYN_T::CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&control_noise_d_,
                            sizeof(float) * DYN_T::CONTROL_DIM *
                            this->num_timesteps_ * NUM_ROLLOUTS * 2));
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::deallocateCUDAMemory() {
    cudaFree(control_d_);
    cudaFree(state_d_);
    cudaFree(trajectory_costs_d_);
    cudaFree(this->control_variance_d_);
    cudaFree(control_noise_d_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::initDDP(const StateCostWeight& q_mat,
                       const Hessian& q_f_mat,
                       const ControlCostWeight& r_mat) {
    util::DefaultLogger logger;
    bool verbose = false;
    ddp_model_  = std::make_shared<ModelWrapperDDP<DYN_T>>(this->model_);
    ddp_solver_ = std::make_shared< DDP<ModelWrapperDDP<DYN_T>>>(dt_,
            this->num_timesteps_, 1, &logger, verbose);
    Q_ = q_mat;
    Qf_ = q_f_mat;
    R_ = r_mat;

    for (int i = 0; i < DYN_T::CONTROL_DIM; i++) {
        control_min_(i) = this->model_->control_rngs_[i].x;
        control_max_(i) = this->model_->control_rngs_[i].y;
    }

    run_cost_ = std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(Q_,
        R_, this->num_timesteps_);
    terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(Qf_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void TubeMPPI::computeFeedbackGains(const state_array& state) {
  Eigen::MatrixXf control_traj = Eigen::MatrixXf::Zero(DYN_T::CONTROL_DIM,
                                                       this->num_timesteps_);
  // replace with transpose?
  for (int t = 0; t < this->num_timesteps_; t++){
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++){
      control_traj(i,t) = nominal_control_[DYN_T::CONTROL_DIM*t + i];
    }
  }
  run_cost_->setTargets(nominal_state_.data(), nominal_control_.data(),
    this->num_timesteps_);
  // Convert state_array to eigen
  Eigen::Matrix<float, DYN_T::STATE_DIM, 1> s;
  for (int i = 0; i < DYN_T::STATE_DIM; i++) {
    s(i) = state[i];
  }
  terminal_cost_->xf = run_cost_->traj_target_x_.col(this->num_timesteps_ - 1);
  result_ = ddp_solver_->run(s, control_traj,
                             *ddp_model_, *run_cost_, *terminal_cost_,
                             control_min_, control_max_);
}
