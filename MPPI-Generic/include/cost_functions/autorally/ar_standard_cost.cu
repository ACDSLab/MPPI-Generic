#include <cost_functions/autorally/ar_standard_cost.cuh>

ARStandardCost::ARStandardCost(cudaStream_t stream) {

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
  paramsToDevice();
}

void ARStandardCost::freeCudaMem() {
  cudaFree(cost_device_);
}

void ARStandardCost::paramsToDevice() {
  HANDLE_ERROR( cudaMemcpyAsync(&cost_device_->params_, &params_, sizeof(ARStandardCostParams), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaMemcpyAsync(&cost_device_->width_, &width_, sizeof(float), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaMemcpyAsync(&cost_device_->height_, &height_, sizeof(float), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaStreamSynchronize(stream_));
}

void ARStandardCost::clearCostmapCPU(int width, int height) {
  if(width > 0 && height > 0) {
    width_ = width;
    height_ = height;
  }

  if(width_ < 0 || height_ < 0) {
    std::cerr << "ERROR: cannot clear costmap on the CPU with size less than 0" << std::endl;
    return;
  }
  track_costs_.clear();
  track_costs_.resize(width_ * height_);

  for (int i = 0; i < width_*height_; i++){
    track_costs_[i].x = 0;
    track_costs_[i].y = 0;
    track_costs_[i].z = 0;
    track_costs_[i].w = 0;
  }
}

std::vector<float4> ARStandardCost::loadTrackData(std::string map_path, Eigen::Matrix3f &R, Eigen::Array3f &trs) {
  // check if file exists
  if(!fileExists(map_path)) {
    std::cerr << "ERROR: map path invalid, " << map_path << std::endl;
    return std::vector<float4>();
  }

  // load the npz file

  // init Costmap

  // copy the track data into CPU side storage
  return std::vector<float4>();
}
