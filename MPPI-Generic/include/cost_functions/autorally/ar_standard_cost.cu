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

bool ARStandardCost::changeCostmapSize(int width, int height) {
  if(height < 0 && width < 0) {
    std::cerr << "ERROR: cannot resize costmap to size less than 1" << std::endl;
    return false;
  }
  if(height != height_ || width != width_) {
    track_costs_.resize(width * height);

    // TODO reallocate GPU memory for texture
  }

  width_ = width;
  height_ = height;
  return true;
}

void ARStandardCost::clearCostmapCPU(int width, int height) {
  changeCostmapSize(width, height);

  if(width_ < 0 && height_ < 0) {
    return;
  }

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
  cnpy::npz_t map_dict = cnpy::npz_load(map_path);
  float x_min, x_max, y_min, y_max, ppm;
  float* xBounds = map_dict["xBounds"].data<float>();
  float* yBounds = map_dict["yBounds"].data<float>();
  float* pixelsPerMeter = map_dict["pixelsPerMeter"].data<float>();
  x_min = xBounds[0];
  x_max = xBounds[1];
  y_min = yBounds[0];
  y_max = yBounds[1];
  ppm = pixelsPerMeter[0];

  int width = int((x_max - x_min)*ppm);
  int height = int((y_max - y_min)*ppm);

  if(!changeCostmapSize(width, height)) {
    std::cerr << "ERROR: load track has invalid sizes" << std::endl;
    return std::vector<float4>();
  }

  std::vector<float4> track_costs(width_*height_);

  float* channel0 = map_dict["channel0"].data<float>();
  float* channel1 = map_dict["channel1"].data<float>();
  float* channel2 = map_dict["channel2"].data<float>();
  float* channel3 = map_dict["channel3"].data<float>();

  // copy the track data into CPU side storage
  for (int i = 0; i < width_*height_; i++){
    track_costs[i].x = channel0[i];
    track_costs[i].y = channel1[i];
    track_costs[i].z = channel2[i];
    track_costs[i].w = channel3[i];
  }

  //Save the scaling and offset
  R << 1./(x_max - x_min), 0,                  0,
          0,                  1./(y_max - y_min), 0,
          0,                  0,                  1;
  trs << -x_min/(x_max - x_min), -y_min/(y_max - y_min), 1;

  return track_costs;
}
