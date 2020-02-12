#include "mppi_core/mppi_controller.cuh"

#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, NUM_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost) : model_(model), cost_(cost) {
    // Call the GPU setup functions of the model and cost
    model_->GPUSetup();
    cost_->GPUSetup();

    // Allocate CUDA memory for the controller
    allocateCUDAMemory();
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
VanillaMPPI::~VanillaMPPIController() {
    // Free the CUDA memory of every object
    model_->freeCudaMem();
    cost_->freeCudaMem();

    // Free the CUDA memory of the controller
    deallocateCUDAMemory();
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::computeControl(state_array state) {

}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::allocateCUDAMemory() {
    HANDLE_ERROR(cudaMalloc((void**)&nominal_control_d_, sizeof(float)*DYN_T::CONTROL_DIM*NUM_TIMESTEPS));
    HANDLE_ERROR(cudaMalloc((void**)&nominal_state_d_, sizeof(float)*DYN_T::STATE_DIM*NUM_TIMESTEPS));
    HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d_, sizeof(float)*NUM_ROLLOUTS));
    HANDLE_ERROR(cudaMalloc((void**)&control_variance_d_, sizeof(float)*DYN_T::CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&control_noise_d_, sizeof(float)*DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS));
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::deallocateCUDAMemory() {
    cudaFree(nominal_control_d_);
    cudaFree(nominal_state_d_);
    cudaFree(trajectory_costs_d_);
    cudaFree(control_variance_d_);
    cudaFree(control_noise_d_);
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void
VanillaMPPI::setCUDAStream(cudaStream_t stream) {
    stream_ = stream;
    model_->bindToStream(stream);
    cost_->bindToStream(stream);
    curandSetStream(gen_, stream); // requires the generator to be created!
}




#undef VanillaMPPI