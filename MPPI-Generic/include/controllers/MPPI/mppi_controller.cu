#include <controllers/MPPI/mppi_controller.cuh>
#include <mppi_core/mppi_common.cuh>

#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost,
                                   float dt,
                                   int max_iter,
                                   float gamma,
                                   int num_timesteps,
                                   const control_array& control_variance,
                                   const control_trajectory& init_control_traj,
                                   cudaStream_t stream) :
dt_(dt), num_iters_(max_iter), gamma_(gamma), stream_(stream) {
    this->model_ = model;
    this->cost_ = cost;

    control_variance_ = control_variance;
    nominal_control_ = init_control_traj;
    setNumTimesteps(num_timesteps);

    // Create the random number generator
    createAndSeedCUDARandomNumberGen();

    // Bind the model and control to the given stream
    setCUDAStream(stream);

    // Call the GPU setup functions of the model and cost
    this->model_->GPUSetup();
    this->cost_->GPUSetup();


    // Allocate CUDA memory for the controller
    allocateCUDAMemory();

    // Copy the noise variance to the device
    copyControlVarianceToDevice();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
VanillaMPPI::~VanillaMPPIController() {
    // Free the CUDA memory of every object
    this->model_->freeCudaMem();
    this->cost_->freeCudaMem();

    // Free the CUDA memory of the controller
    deallocateCUDAMemory();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::computeControl(const state_array& state) {

    // Send the initial condition to the device
    HANDLE_ERROR( cudaMemcpyAsync(initial_state_d_, state.data(),
        DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));

    for (int opt_iter = 0; opt_iter < num_iters_; opt_iter++) {
        // Send the nominal control to the device
        copyNominalControlToDevice();

        //Generate noise data
        curandGenerateNormal(gen_, control_noise_d_,
                             NUM_ROLLOUTS*num_timesteps_*DYN_T::CONTROL_DIM,
                             0.0, 1.0);

        //Launch the rollout kernel
        mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
            this->model_->model_d_, this->cost_->cost_d_, dt_, num_timesteps_,
            initial_state_d_, nominal_control_d_, control_noise_d_,
            control_variance_d_, trajectory_costs_d_, stream_);

        // Copy the costs back to the host
        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_.data(),
            trajectory_costs_d_,
            NUM_ROLLOUTS*sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR( cudaStreamSynchronize(stream_) );

        baseline_ = mppi_common::computeBaselineCost(trajectory_costs_.data(),
            NUM_ROLLOUTS);

        // Launch the norm exponential kernel
        mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X,
            trajectory_costs_d_, gamma_, baseline_, stream_);
        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_.data(),
            trajectory_costs_d_,
            NUM_ROLLOUTS*sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR(cudaStreamSynchronize(stream_));

        // Compute the normalizer
        normalizer_ = mppi_common::computeNormalizer(trajectory_costs_.data(),
            NUM_ROLLOUTS);

        // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
        mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
            trajectory_costs_d_, control_noise_d_, nominal_control_d_,
            normalizer_, num_timesteps_, stream_);

        // Transfer the new control to the host
        HANDLE_ERROR( cudaMemcpyAsync(nominal_control_.data(), nominal_control_d_,
                sizeof(float)*num_timesteps_*DYN_T::CONTROL_DIM,
                cudaMemcpyDeviceToHost, stream_));
        cudaStreamSynchronize(stream_);

        // TODO Add SavitskyGolay?
        // TODO Add nominal state computation
        computeNominalStateTrajectory(state);
    }

}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::allocateCUDAMemory() {
    HANDLE_ERROR(cudaMalloc((void**)&initial_state_d_,
                            sizeof(float)*DYN_T::STATE_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&nominal_control_d_,
                            sizeof(float)*DYN_T::CONTROL_DIM*num_timesteps_));
    HANDLE_ERROR(cudaMalloc((void**)&nominal_state_d_,
                            sizeof(float)*DYN_T::STATE_DIM*num_timesteps_));
    HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d_,
                            sizeof(float)*NUM_ROLLOUTS));
    HANDLE_ERROR(cudaMalloc((void**)&control_variance_d_,
                            sizeof(float)*DYN_T::CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&control_noise_d_,
                            sizeof(float)*DYN_T::CONTROL_DIM*num_timesteps_*NUM_ROLLOUTS));
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::deallocateCUDAMemory() {
    cudaFree(nominal_control_d_);
    cudaFree(nominal_state_d_);
    cudaFree(trajectory_costs_d_);
    cudaFree(control_variance_d_);
    cudaFree(control_noise_d_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void
VanillaMPPI::setCUDAStream(cudaStream_t stream) {
    stream_ = stream;
    this->model_->bindToStream(stream);
    this->cost_->bindToStream(stream);
    curandSetStream(gen_, stream); // requires the generator to be created!
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::updateControlNoiseVariance(const control_array &sigma_u) {
    control_variance_ = sigma_u;
    copyControlVarianceToDevice();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::copyControlVarianceToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(control_variance_d_, control_variance_.data(), sizeof(float)*control_variance_.size(), cudaMemcpyHostToDevice, stream_));
    cudaStreamSynchronize(stream_);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::copyNominalControlToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(nominal_control_d_, nominal_control_.data(), sizeof(float)*nominal_control_.size(), cudaMemcpyHostToDevice, stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::computeNominalStateTrajectory(const state_array &x0) {
  nominal_state_.col(0) = x0;
  state_array xdot;
//  for (int i =0; i < num_timesteps_ - 1; ++i) {
//    nominal_state_.col(i+1) = nominal_state_.col(i);
//    state_array state = nominal_state_.col(i+1);
//    control_array control = nominal_control_.col(i);
//    this->model_->computeStateDeriv(state, control, xdot);
//    this->model_->updateState(state, xdot, dt_);
//    nominal_state_.col(i+1) = state;
//    }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void
VanillaMPPI::setNumTimesteps(int num_timesteps) {
    if ((num_timesteps <= MAX_TIMESTEPS) && (num_timesteps > 0)) {
        num_timesteps_ = num_timesteps;
    } else {
        num_timesteps_ = MAX_TIMESTEPS;
        printf("You must give a number of timesteps between [0, %d]\n", MAX_TIMESTEPS);
    }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void
VanillaMPPI::createAndSeedCUDARandomNumberGen() {
    // Seed the PseudoRandomGenerator with the CPU time.
    curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    curandSetPseudoRandomGeneratorSeed(gen_, seed);
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
void VanillaMPPI::slideControlSequence(int steps) {
    for (int i = 0; i < num_timesteps_; ++i) {
        for (int j = 0; j < DYN_T::CONTROL_DIM; j++) {
            if (i + steps < num_timesteps_) {
                nominal_control_(j,i) = nominal_control_(j,i + steps);
            } else {
                nominal_control_(j,i) = nominal_control_(j,num_timesteps_-1);
            }
        }
    }
}

//template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
//void
//VanillaMPPI::computeNominalStateTrajectory(const state_array& x0) {
//    // Increment the system forward
//    for (int i = 0; i < DYN_T::STATE_DIM; i++) {
//        nominal_state_[i] = x0[i];
//    }
//    for (int i = 1; i < NUM_TIMESTEPS; i++) {
//        for (int j = 0; j < DYN_T::STATE_DIM; j++) {
//            nominal_state_[i*DYN_T::STATE_DIM + j] = model_
//        }
//
//    }
//
//}


#undef VanillaMPPI
