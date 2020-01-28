#include <cost_functions/cartpole/cartpole_quadratic_cost.cuh>

CartPoleQuadraticCost::CartPoleQuadraticCost(cudaStream_t stream) {
    bindToStream(stream);
}

CartPoleQuadraticCost::~CartPoleQuadraticCost() {
    if (!GPUMemStatus_) {
        freeCudaMem();
    }
}

void CartPoleQuadraticCost::GPUSetup() {
    if (!GPUMemStatus_) {
        cost_d_ = Managed::GPUSetup(this);
    } else {
        std::cout << "GPU Memory already set" << std::endl; //TODO should this be an exception?
    }
    paramsToDevice();
}

void CartPoleQuadraticCost::freeCudaMem() {
    cudaFree(cost_d_);
}

void CartPoleQuadraticCost::setParams(CartPoleQuadraticCostParams params) {
    this->params_ = params;
    if(GPUMemStatus_) {
        paramsToDevice();
    }
}

void CartPoleQuadraticCost::paramsToDevice() {
    HANDLE_ERROR( cudaMemcpyAsync(&cost_d_->params_, &params_, sizeof(CartPoleQuadraticCostParams), cudaMemcpyHostToDevice, stream_));
    HANDLE_ERROR( cudaStreamSynchronize(stream_));
}

__host__ __device__ float CartPoleQuadraticCost::getStateCost(float *state) {
    return state[0]*state[0]*params_.cart_position_coeff +
           state[1]*state[1]*params_.cart_velocity_coeff +
           state[2]*state[2]*params_.pole_angle_coeff +
           state[3]*state[3]*params_.pole_angular_velocity_coeff;
}

__host__ __device__ float CartPoleQuadraticCost::getControlCost(float *u, float *du, float *vars) {
    return params_.control_force_coeff*du[0]*(u[0] - du[0])/(vars[0]*vars[0]);
}

__host__ __device__ float CartPoleQuadraticCost::computeRunningCost(float *s, float *u, float *du, float *vars) {
    return getStateCost(s) + getControlCost(u, du, vars);
}

