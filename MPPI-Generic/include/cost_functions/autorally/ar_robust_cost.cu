
template<class CLASS_T, class PARAMS_T>
ARRobustCost<CLASS_T, PARAMS_T>::ARRobustCost(cudaStream_t stream) : ARStandardCost<ARRobustCost<CLASS_T, PARAMS_T>, PARAMS_T>(stream) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

template<class CLASS_T, class PARAMS_T>
ARRobustCost<CLASS_T, PARAMS_T>::~ARRobustCost() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

template <class CLASS_T, class PARAMS_T>
void ARRobustCost<CLASS_T, PARAMS_T>::GPUSetup() {
  //std::cout << __PRETTY_FUNCTION__ << std::endl;
  if (!this->GPUMemStatus_) {
    this->cost_d_ = Managed::GPUSetup(this);
  } else {
    std::cout << "GPU Memory already set." << std::endl;
  }
  // load track data
  // update transform
  // update params
  // allocate texture memory
  // convert costmap to texture
  this->paramsToDevice();
}

template<class CLASS_T, class PARAMS_T>
__host__ __device__ float ARRobustCost<CLASS_T, PARAMS_T>::getStabilizingCost(float* s)
{
  float penalty_val = 0;
  float slip;
  if (fabs(s[4]) < 0.25){
    slip = 0;
  } else {
    slip = fabs(-atan(s[5]/fabs(s[4])));
  }
  if (slip >= this->params_.max_slip_ang){
    penalty_val = this->params_.crash_coeff;
  }
  // crash if roll is too large
  if (fabs(s[3]) >= 1.5){
    penalty_val += this->params_.crash_coeff;
  }
  return this->params_.slip_coeff*slip + penalty_val;
}

template<class CLASS_T, class PARAMS_T>
__device__ float ARRobustCost<CLASS_T, PARAMS_T>::getCostmapCost(float* s)
{
  float cost = 0;

  //Compute a transformation to get the (x,y) positions of the front and back of the car.
  float x_front = s[0] + this->FRONT_D*__cosf(s[2]);
  float y_front = s[1] + this->FRONT_D*__sinf(s[2]);
  float x_back = s[0] + this->BACK_D*__cosf(s[2]);
  float y_back = s[1] + this->BACK_D*__sinf(s[2]);

  float u,v,w; //Transformed coordinates

  // parameters for the front and back of car
  this->coorTransform(x_front, y_front, &u, &v, &w);
  float4 track_params_front = tex2D<float4>(this->costmap_tex_d_, u/w, v/w);
  this->coorTransform(x_back, y_back, &u, &v, &w);
  float4 track_params_back = tex2D<float4>(this->costmap_tex_d_, u/w, v/w);


  //Calculate the constraint penalty
  float constraint_val = fminf(1.0, fmaxf(track_params_front.x, track_params_back.x));

  if (constraint_val >= this->params_.boundary_threshold) {
    float alpha = (constraint_val - this->params_.boundary_threshold)/(1.0 - this->params_.boundary_threshold);
    cost += alpha*this->params_.crash_coeff;
  }

  //Calculate the track positioning cost
  if (track_params_front.y > this->params_.track_slop){
    cost += this->params_.track_coeff*track_params_front.y;
  }

  //Calculate the speed cost
  if(this->params_.desired_speed == -1) {
    cost += this->params_.speed_coeff*fabs(s[4] - track_params_front.z);
  } else {
    cost += this->params_.speed_coeff*fabs(this->params_.desired_speed - track_params_front.z);
  }

  //Calculate the heading cost
  cost += this->params_.heading_coeff*fabs(sinf(s[2]) + track_params_front.w);

  return cost;
}

//Compute the immediate running cost.
template<class CLASS_T, class PARAMS_T>
__device__ float ARRobustCost<CLASS_T, PARAMS_T>::computeCost(float* s, float* u, float* du,
                                           float* vars, int* crash, int timestep)
{
  float control_cost = this->getControlCost(u, du, vars);
  float stabilizing_cost = getStabilizingCost(s);
  float costmap_cost = getCostmapCost(s, timestep);
  float cost = stabilizing_cost + control_cost + costmap_cost;
  if (cost > this->MAX_COST_VALUE || isnan(cost)) {
    cost = this->MAX_COST_VALUE;
  }
  return cost;
}
