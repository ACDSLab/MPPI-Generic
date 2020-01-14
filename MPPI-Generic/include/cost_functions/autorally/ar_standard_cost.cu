#include <cost_functions/autorally/ar_standard_cost.cuh>

ARStandardCost::ARStandardCost(cudaStream_t stream) {

  bindToStream(stream);
}

ARStandardCost::~ARStandardCost() {

}

void ARStandardCost::setParams(ARStandardCostParams params) {
  this->params_ = params;
  if(GPUMemStatus_) {
    paramsToDevice();
  }
}

void ARStandardCost::GPUSetup() {
  if (!GPUMemStatus_) {
    cost_d_ = Managed::GPUSetup(this);
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
  // TODO free everything
  //cudaFree();
  cudaFree(cost_d_);
}

void ARStandardCost::paramsToDevice() {
  HANDLE_ERROR( cudaMemcpyAsync(&cost_d_->params_, &params_, sizeof(ARStandardCostParams), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaMemcpyAsync(&cost_d_->width_, &width_, sizeof(float), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaMemcpyAsync(&cost_d_->height_, &height_, sizeof(float), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaStreamSynchronize(stream_));
}

bool ARStandardCost::changeCostmapSize(int width, int height) {
  // TODO set flag at top that indicates memory allocation changes
  if(height < 0 && width < 0) {
    std::cerr << "ERROR: cannot resize costmap to size less than 1" << std::endl;
    return false;
  }
  if(height != height_ || width != width_) {
    track_costs_.resize(width * height);

    //Allocate memory for the cuda array which is bound the costmap_tex_
    // has been allocated in the past, must be freed
    if(height_ > 0 && width_ > 0) {
      HANDLE_ERROR(cudaFreeArray(costmapArray_d_));
    }
    // 4 floats of size 32 bits
    channelDesc_ = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    HANDLE_ERROR(cudaMallocArray(&costmapArray_d_, &channelDesc_, width, height));

    // set all of the elements in the array to be zero
    std::vector<float4> zero_array(width_*height_);
    zero_array.resize(0, make_float4(0,0,0,0));
    HANDLE_ERROR(cudaMemcpyToArray(costmapArray_d_, 0, 0, zero_array.data(), width*height*sizeof(float4), cudaMemcpyHostToDevice));
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

  float* channel0 = map_dict["channel0"].data<float>();
  float* channel1 = map_dict["channel1"].data<float>();
  float* channel2 = map_dict["channel2"].data<float>();
  float* channel3 = map_dict["channel3"].data<float>();

  // copy the track data into CPU side storage
  for (int i = 0; i < width_*height_; i++){
    track_costs_[i].x = channel0[i];
    track_costs_[i].y = channel1[i];
    track_costs_[i].z = channel2[i];
    track_costs_[i].w = channel3[i];
  }

  //Save the scaling and offset
  R << 1./(x_max - x_min), 0,                  0,
          0,                  1./(y_max - y_min), 0,
          0,                  0,                  1;
  trs << -x_min/(x_max - x_min), -y_min/(y_max - y_min), 1;

  return track_costs_;
}

void ARStandardCost::costmapToTexture() {
  if(width_ < 0 || height_ < 0) {
    std::cerr << "ERROR: cannot allocate texture with zero size" << std::endl;
    return;
  }

  // transfer CPU version of costmap to GPU
  float4* costmap_ptr = track_costs_.data();
  HANDLE_ERROR(cudaMemcpyToArray(costmapArray_d_ , 0, 0, costmap_ptr, width_*height_*sizeof(float4), cudaMemcpyHostToDevice));
  cudaStreamSynchronize(stream_);

  //Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = costmapArray_d_;

  //Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  //Destroy current texture and create new texture object
  HANDLE_ERROR(cudaDestroyTextureObject(costmap_tex_d_));
  HANDLE_ERROR(cudaCreateTextureObject(&costmap_tex_d_, &resDesc, &texDesc, NULL) );

  // copy over pointers setup up on CPU code to GPU
  HANDLE_ERROR( cudaMemcpyAsync(&cost_d_->costmapArray_d_, &costmapArray_d_, sizeof(cudaArray*), cudaMemcpyHostToDevice, stream_));
  HANDLE_ERROR( cudaMemcpyAsync(&cost_d_->costmap_tex_d_, &costmap_tex_d_, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice, stream_));
  cudaStreamSynchronize(stream_);
}

inline __device__ float4 ARStandardCost::queryTexture(float x, float y) const {
  printf("\nquerying point (%f, %f)", x, y);
  return tex2D<float4>(costmap_tex_d_, x, y);
}

void ARStandardCost::updateTransform(Eigen::MatrixXf m, Eigen::ArrayXf trs) {
  params_.r_c1.x = m(0,0);
  params_.r_c1.y = m(1,0);
  params_.r_c1.z = m(2,0);
  params_.r_c2.x = m(0,1);
  params_.r_c2.y = m(1,1);
  params_.r_c2.z = m(2,1);
  params_.trs.x = trs(0);
  params_.trs.y = trs(1);
  params_.trs.z = trs(2);
  //Move the updated parameters to gpu memory
  if(GPUMemStatus_) {
    paramsToDevice();
  }
}

__host__ __device__ void ARStandardCost::coorTransform(float x, float y, float* u, float* v, float* w) {
  //Compute a projective transform of (x, y, 0, 1)
  u[0] = params_.r_c1.x*x + params_.r_c2.x*y + params_.trs.x;
  v[0] = params_.r_c1.y*x + params_.r_c2.y*y + params_.trs.y;
  w[0] = params_.r_c1.z*x + params_.r_c2.z*y + params_.trs.z;
}
