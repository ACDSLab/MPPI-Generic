#include <cost_functions/autorally/ar_standard_cost.cuh>

ARStandardCost::ARStandardCost(int width, int height, cudaStream_t stream) {
  this->width_ = width;
  this->height_ = height;

  bindToStream(stream);
}

ARStandardCost::~ARStandardCost() {

}

void ARStandardCost::GPUSetup() {
  if (!GPUMemStatus_) {
    cost_device_ = Managed::GPUSetup(this);
  } else {
    std::cout << "GPU Memory already set." << std::endl;
  }
  // load track data
  // update transform
  // update params
  // allocate texture memory
  // convert costmap to texture
}

void ARStandardCost::freeCudaMem() {
  cudaFree(cost_device_);
}

void ARStandardCost::paramsToDevice() {
  HANDLE_ERROR( cudaMemcpyAsync(&cost_device_->params_, &params_, sizeof(float), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaStreamSynchronize(stream_));
}

