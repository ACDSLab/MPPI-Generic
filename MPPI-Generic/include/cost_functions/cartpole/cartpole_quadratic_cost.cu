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

