#include <cost_functions/autorally/ar_standard_cost.cuh>

ARStandardCost::ARStandardCost(cudaStream_t stream) {

  bindToStream(stream);
}

ARStandardCost::~ARStandardCost() {
  if(GPUMemStatus_) {
    freeCudaMem();
  }
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

std::vector<float4> ARStandardCost::loadTrackData(std::string map_path) {
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

  Eigen::Matrix3f R;
  Eigen::Array3f trs;

  //Save the scaling and offset
  R << 1./(x_max - x_min), 0,                  0,
          0,                  1./(y_max - y_min), 0,
          0,                  0,                  1;
  trs << -x_min/(x_max - x_min), -y_min/(y_max - y_min), 1;

  updateTransform(R, trs);
  costmapToTexture();

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
  //printf("\nquerying point (%f, %f)", x, y);
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
  ////Compute a projective transform of (x, y, 0, 1)
  //printf("coordiante transform %f, %f, %f\n", params_.r_c1.x, params_.r_c2.x, params_.trs.x);
  u[0] = params_.r_c1.x*x + params_.r_c2.x*y + params_.trs.x;
  v[0] = params_.r_c1.y*x + params_.r_c2.y*y + params_.trs.y;
  w[0] = params_.r_c1.z*x + params_.r_c2.z*y + params_.trs.z;
}

__device__ float4 ARStandardCost::queryTextureTransformed(float x, float y) {
  float u, v, w;
  coorTransform(x, y, &u, &v, &w);
  //printf("\ninput coordinates: %f, %f", x, y);
  //printf("\nu = %f, v = %f, w = %f", u, v, w);
  //printf("\ntransformed coordinates %f, %f\n", u/w, v/w);
  return tex2D<float4>(costmap_tex_d_, u/w, v/w);
}

Eigen::Matrix3f ARStandardCost::getRotation() {
  Eigen::Matrix3f m;
  m(0,0) = params_.r_c1.x;
  m(1,0) = params_.r_c1.y;
  m(2,0) = params_.r_c1.z;
  m(0,1) = params_.r_c2.x;
  m(1,1) = params_.r_c2.y;
  m(2,1) = params_.r_c2.z;
  m(0,2) = 0.0;
  m(1,2) = 0.0;
  m(2,2) = 1.0;
  return m;
}

Eigen::Array3f ARStandardCost::getTranslation() {
  Eigen::Array3f array;
  array(0) = params_.trs.x;
  array(1) = params_.trs.y;
  array(2) = params_.trs.z;
  return array;
}

inline __host__ __device__ float ARStandardCost::getTerminalCost(float *s) {
  return 0.0;
}

inline __host__ __device__ float ARStandardCost::getControlCost(float *u, float *du, float *vars) {
  float control_cost = 0.0;
  //printf("du %f, %f\n", du[0], du[1]);
//printf("vars %f, %f\n", vars[0], vars[1]);
  //printf("vars %f, %f\n", u[0], u[1]);
  control_cost += params_.steering_coeff*du[0]*(u[0] - du[0])/(vars[0]*vars[0]);
  control_cost += params_.throttle_coeff*du[1]*(u[1] - du[1])/(vars[1]*vars[1]);
  return control_cost;
}

inline __host__ __device__ float ARStandardCost::getSpeedCost(float *s, int *crash) {
  float cost = 0;
  float error = s[4] - params_.desired_speed;
  if (l1_cost_){
    cost = fabs(error);
  }
  else {
    cost = error*error;
  }
  return (params_.speed_coeff*cost);
}

inline __host__ __device__ float ARStandardCost::getStabilizingCost(float *s) {
  float stabilizing_cost = 0;
  if (fabs(s[4]) > 0.001) {
    float slip = -atan(s[5]/fabs(s[4]));
    stabilizing_cost = params_.slip_penalty*powf(slip,2);
    if (fabs(-atan(s[5]/fabs(s[4]))) > params_.max_slip_ang) {
      //If the slip angle is above the max slip angle kill the trajectory.
      stabilizing_cost += params_.crash_coeff;
    }
  }
  return stabilizing_cost;
}

inline __host__ __device__ float ARStandardCost::getCrashCost(float *s, int *crash, int num_timestep) {
  float crash_cost = 0;
  if (crash[0] > 0) {
    crash_cost = params_.crash_coeff;
  }
  return crash_cost;
}

inline __device__ float ARStandardCost::getTrackCost(float *s, int *crash) {
  float track_cost = 0;

  //Compute a transformation to get the (x,y) positions of the front and back of the car.
  float x_front = s[0] + FRONT_D*__cosf(s[2]);
  float y_front = s[1] + FRONT_D*__sinf(s[2]);
  float x_back = s[0] + BACK_D*__cosf(s[2]);
  float y_back = s[1] + BACK_D*__sinf(s[2]);

  //Cost of front of the car
  float track_cost_front = queryTextureTransformed(x_front, y_front).x;
  //Cost for back of the car
  float track_cost_back = queryTextureTransformed(x_back, y_back).x;

  track_cost = (fabs(track_cost_front) + fabs(track_cost_back) )/2.0;
  if (fabs(track_cost) < params_.track_slop) {
    track_cost = 0;
  }
  else {
    track_cost = params_.track_coeff*track_cost;
  }
  if (track_cost_front >= params_.boundary_threshold || track_cost_back >= params_.boundary_threshold) {
    crash[0] = 1;
  }
  return track_cost;
}

inline __device__ float ARStandardCost::computeCost(float *s, float *u, float *du, float *vars, int *crash, int timestep) {
  float control_cost = getControlCost(u, du, vars);
  float track_cost = getTrackCost(s, crash);
  float speed_cost = getSpeedCost(s, crash);
  float crash_cost = powf(params_.discount, timestep)*getCrashCost(s, crash, timestep);
  float stabilizing_cost = getStabilizingCost(s);
  float cost = control_cost + speed_cost + crash_cost + track_cost + stabilizing_cost;
  if (cost > MAX_COST_VALUE || isnan(cost)) {
    cost = MAX_COST_VALUE;
  }
  return cost;
}
