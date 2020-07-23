#include <cnpy.h>
#include <mppi/utils/file_utils.h>
#include <mppi/cost_functions/quadrotor/quadrotor_map_cost.cuh>

template <class CLASS_T, class PARAMS_T>
QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::QuadrotorMapCostImpl(cudaStream_t stream) {
  this->bindToStream(stream);
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::freeCudaMem() {
  if (this->GPUMemStatus_) {
    HANDLE_ERROR(cudaFreeArray(costmapArray_d_));
    HANDLE_ERROR(cudaDestroyTextureObject(costmap_tex_d_));
  }
  Cost<CLASS_T, PARAMS_T, this->STATE_DIM, this->CONTROL_DIM>::freeCudaMem();
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::paramsToDevice() {
  if (this->GPUMemStatus_) {
    HANDLE_ERROR( cudaMemcpyAsync(&this->cost_d_->params_, &this->params_, sizeof(PARAMS_T), cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR( cudaMemcpyAsync(&this->cost_d_->width_, &width_, sizeof(float), cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR( cudaMemcpyAsync(&this->cost_d_->height_, &height_, sizeof(float), cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR( cudaStreamSynchronize(this->stream_));
  }
}

template <class CLASS_T, class PARAMS_T>
float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    const Eigen::Ref<const state_array> s, int timestep, int* crash_status) {
  // TODO query texture on CPU
  float cost = 0;
  cost += computeGateSideCost(s.data());
  cost += computeHeightCost(s.data());
  cost += computeHeadingCost(s.data());
  cost += computeSpeedCost(s.data());
  cost += computeStabilizingCost(s.data());

  // Decrease cost if we pass a gate
  float dist_to_gate = distToWaypoint(s.data(), this->params_.curr_waypoint);

  if (dist_to_gate < this->params_.gate_margin) {
    cost += this->params_.gate_pass_cost;
  }
  return cost;
}

template <class CLASS_T, class PARAMS_T>
__device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    float* s, int timestep, int* crash_status) {
  float cost = 0;
  cost += computeCostmapCost(s);
  cost += computeGateSideCost(s);
  cost += computeHeightCost(s);
  cost += computeHeadingCost(s);
  cost += computeSpeedCost(s);
  cost += computeStabilizingCost(s);

  // Decrease cost if we pass a gate
  float dist_to_gate = distToWaypoint(s, this->params_.curr_waypoint);

  if (dist_to_gate < this->params_.gate_margin) {
    cost += this->params_.gate_pass_cost;
  }
  return cost;
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::distToWaypoint(float* s,
    float4 waypoint) {
  float dist = sqrt(powf(s[0] - waypoint.x, 2) +
                    powf(s[1] - waypoint.y, 2) +
                    powf(s[2] - waypoint.z, 2));

  return dist;
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::updateWaypoint(float4 new_waypoint) {
  updateWaypoint(new_waypoint.x, new_waypoint.y, new_waypoint.z, new_waypoint.w);
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::updateWaypoint(float x, float y,
                                                             float z,
                                                             float heading) {
  if (this->params_.updateWaypoint(x, y, z, heading)) {
    paramsToDevice();
  }
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::updateGateBoundaries(float3 left_side,
                                                                   float3 right_side) {
  updateGateBoundaries(left_side.x, left_side.y, left_side.z,
                       right_side.x, right_side.y, right_side.z);
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::updateGateBoundaries(
    std::vector<float> boundaries) {
  if (boundaries.size() < 6) {
    std::cerr << "You need " << 6 - boundaries.size() << " more floats in the"
              << " call to updateGateBoundaries" << std::endl;
    return;
  }
  updateGateBoundaries(boundaries[0], boundaries[1], boundaries[2],
                       boundaries[3], boundaries[4], boundaries[5]);
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::updateGateBoundaries(float left_x,
                                                                   float left_y,
                                                                   float left_z,
                                                                   float right_x,
                                                                   float right_y,
                                                                   float right_z) {
  if (this->params_.updateGateBoundaries(left_x, left_y, left_z,
                                         right_x, right_y, right_z)) {
    paramsToDevice();
  }
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeStabilizingCost(
    float* s) {
  float cost = 0;
  float roll, pitch, yaw;
  mppi_math::Quat2EulerNWU(&s[6], roll, pitch, yaw);

  float quat_dist_from_level = powf(roll, 2) + powf(pitch, 2);
  cost += this->params_.attitude_coeff * quat_dist_from_level;
  return cost;
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeHeadingCost(
    float* s) {
  float cost = 0;
  float roll, pitch, yaw;
  mppi_math::Quat2EulerNWU(&s[6], roll, pitch, yaw);

  // Calculate heading to gate
  float wx = this->params_.curr_waypoint.x - s[0];
  float wy = this->params_.curr_waypoint.y - s[1];
  float w_heading = atan2f(wy, wx);


  float dist_to_gate = distToWaypoint(s, this->params_.curr_waypoint);
  // Far away from the gate, we want to be pointing at the gate
  if (dist_to_gate > this->params_.gate_margin) {
    cost += this->params_.heading_coeff * powf(yaw - w_heading, 2);
  }
  return cost;
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeSpeedCost(
    float* s) {
  float cost = 0;
  float speed = sqrt(s[3] * s[3] + s[4] * s[4]);
  cost = this->params_.speed_coeff * powf(speed - this->params_.desired_speed, 2);
  return cost;
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeGateSideCost(
    float* s) {
  float cost = 0;
  //Calculate the side border cost
  float dist_to_left_side = sqrtf(powf(s[0] - this->params_.curr_gate_left.x, 2) +
                                  powf(s[1] - this->params_.curr_gate_left.y, 2));
  float dist_to_right_side = sqrtf(powf(s[0] - this->params_.curr_gate_right.x, 2) +
                                   powf(s[1] - this->params_.curr_gate_right.y, 2));
  float prev_dist_to_left_side = sqrtf(powf(s[0] - this->params_.prev_gate_left.x, 2) +
                                       powf(s[1] - this->params_.prev_gate_left.y, 2));
  float prev_dist_to_right_side = sqrtf(powf(s[0] - this->params_.prev_gate_right.x, 2) +
                                        powf(s[1] - this->params_.prev_gate_right.y, 2));

  // Find the side closest to the current state
  float closest_side_dist = fminf(dist_to_left_side, fminf(dist_to_right_side,
      fminf(prev_dist_to_left_side, prev_dist_to_right_side)));
  if (closest_side_dist < this->params_.min_dist_to_gate_side) {
    cost += this->params_.crash_coeff;
  }
  return cost;
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeHeightCost(
    float* s) {

  // Calculate height cost
  float cost = 0;
  float d1 = sqrtf(powf(s[0] - this->params_.prev_waypoint.x, 2) +
	                 powf(s[1] - this->params_.prev_waypoint.y, 2));
  float d2 = sqrtf(powf(s[0] - this->params_.curr_waypoint.x, 2) +
	                 powf(s[1] - this->params_.curr_waypoint.y, 2));
  float w1 = d1 / (d1 + d2);
  float w2 = d2 / (d1 + d2);
  float interpolated_height = (1.0 - w1) * this->params_.prev_waypoint.z +
                              (1.0 - w2) * this->params_.curr_waypoint.z;
  float height_diff = fabs(s[2] - interpolated_height);
  cost += this->params_.height_coeff * height_diff;
  return cost;
}

template <class CLASS_T, class PARAMS_T>
__device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::computeCostmapCost(
    float* s) {
  float cost = 0;

  float u, v, w; // Transformed coordinates
  coorTransform(s[0], s[1], &u, &v, &w);
  float normalized_u = u / w;
  float normalized_v = v / w;

  // Query texture
  float4 track_params = queryTexture(normalized_u, normalized_v);
  // Outside of cost map
  if (normalized_u < 0.001 || normalized_u > 0.999 ||
      normalized_v < 0.001 || normalized_v > 0.999) {
    cost += this->params_.crash_coeff;
  }

  // Calculate cost based on distance from centerline of the track
  if (track_params.x > this->params_.track_slop) {
    cost += this->params_.track_coeff * track_params.x;
  }

  if (track_params.x > this->params_.track_boundary_cost) {
    // the cost at this point on the costmap indicates no longer being on the track
    cost += this->params_.crash_coeff;
  }

  return cost;
}

template <class CLASS_T, class PARAMS_T>
float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::terminalCost(
    const Eigen::Ref<const state_array> s) {
  std::cout << "It is a cost function" << std::endl;
  return 0;
}

template <class CLASS_T, class PARAMS_T>
__device__ float QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::terminalCost(float* s) {
  return 0;
}

template <class CLASS_T, class PARAMS_T>
std::vector<float4> QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::loadTrackData(std::string map_path) {
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
    //std::cout << i << " = " << channel0[i] << ", " << channel1[i] << ", " << channel2[i] << ", " << channel3[i] << std::endl;
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


template <class CLASS_T, class PARAMS_T>
bool QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::changeCostmapSize(
    int width, int height) {
  // TODO set flag at top that indicates memory allocation changes
  if (height < 0 && width < 0) {
    std::cerr << "ERROR: cannot resize costmap to size less than 1" << std::endl;
    return false;
  }
  if (height != height_ || width != width_) {
    track_costs_.resize(width * height);

    //Allocate memory for the cuda array which is bound the costmap_tex_
    // has been allocated in the past, must be freed
    if (height_ > 0 && width_ > 0) {
      HANDLE_ERROR(cudaFreeArray(costmapArray_d_));
    }
    // 4 floats of size 32 bits
    channelDesc_ = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    HANDLE_ERROR(cudaMallocArray(&costmapArray_d_, &channelDesc_, width, height));

    // set all of the elements in the array to be zero
    std::vector<float4> zero_array(width_*height_);
    zero_array.resize(width*height, make_float4(0,0,0,0));
    HANDLE_ERROR(cudaMemcpyToArray(costmapArray_d_, 0, 0, zero_array.data(),
                                   width * height * sizeof(float4),
                                   cudaMemcpyHostToDevice));
  }

  width_ = width;
  height_ = height;
  return true;
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::updateTransform(
    const Eigen::Ref<const Eigen::Matrix3f>& m,
    const Eigen::Ref<const Eigen::Array3f>& trs) {

  this->params_.r_c1.x = m(0,0);
  this->params_.r_c1.y = m(1,0);
  this->params_.r_c1.z = m(2,0);
  this->params_.r_c2.x = m(0,1);
  this->params_.r_c2.y = m(1,1);
  this->params_.r_c2.z = m(2,1);
  this->params_.trs.x = trs(0);
  this->params_.trs.y = trs(1);
  this->params_.trs.z = trs(2);
  //Move the updated parameters to gpu memory
  if(this->GPUMemStatus_) {
    paramsToDevice();
  }
}

template <class CLASS_T, class PARAMS_T>
void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::costmapToTexture() {
  if(width_ < 0 || height_ < 0) {
    std::cerr << "ERROR: cannot allocate texture with zero size" << std::endl;
    return;
  }

  // transfer CPU version of costmap to GPU
  float4* costmap_ptr = track_costs_.data();
  HANDLE_ERROR(cudaMemcpyToArray(costmapArray_d_ , 0, 0, costmap_ptr, width_*height_*sizeof(float4), cudaMemcpyHostToDevice));
  cudaStreamSynchronize(this->stream_);

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
  HANDLE_ERROR( cudaMemcpyAsync(&this->cost_d_->costmapArray_d_, &costmapArray_d_, sizeof(cudaArray*), cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR( cudaMemcpyAsync(&this->cost_d_->costmap_tex_d_, &costmap_tex_d_, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice, this->stream_));
  cudaStreamSynchronize(this->stream_);
}

template <class CLASS_T, class PARAMS_T>
__device__ float4 QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::queryTexture(float x,
                                                                        float y) const {
  return tex2D<float4>(costmap_tex_d_, x, y);
}

template <class CLASS_T, class PARAMS_T>
__device__ float4 QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::queryTextureTransformed(
    float x, float y) {
  float u, v, w;
  coorTransform(x, y, &u, &v, &w);
  return queryTexture(u / w, v / w);
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ void QuadrotorMapCostImpl<CLASS_T, PARAMS_T>::coorTransform(
    float x, float y, float* u, float* v, float* w) {
  ////Compute a projective transform of (x, y, 0, 1)
  // converts to the texture [0-1] coordinate system
  u[0] = this->params_.r_c1.x*x + this->params_.r_c2.x*y + this->params_.trs.x;
  v[0] = this->params_.r_c1.y*x + this->params_.r_c2.y*y + this->params_.trs.y;
  w[0] = this->params_.r_c1.z*x + this->params_.r_c2.z*y + this->params_.trs.z;
}