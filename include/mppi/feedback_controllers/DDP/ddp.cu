#include <mppi/feedback_controllers/DDP/ddp.cuh>

template <class GPU_FB_T, class DYN_T>
DeviceDDPImpl<GPU_FB_T, DYN_T>::DeviceDDPImpl(int num_timesteps, cudaStream_t stream) :
  num_timesteps_(num_timesteps), GPUFeedbackController<DeviceDDPImpl<GPU_FB_T, DYN_T>, DYN_T>(stream) {

}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::allocateCUDAMemory() {
  int fb_size = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * num_timesteps_;
  std::cout << "fb_size: " << fb_size << std::endl;
  std::cout << "feedback_d_: " << this->feedback_d_ << std::endl;
  cudaMalloc((void**)&this->feedback_d_->fb_gains_, fb_size * sizeof(float));
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::deallocateCUDAMemory() {
  if (this->feedback_d_->fb_gains_ != nullptr) {
    cudaFree(this->feedback_d_->fb_gains_);
    this->feedback_d_->fb_gains_ = nullptr;
  }
}

template <class GPU_FB_T, class DYN_T>
__device__ void DeviceDDPImpl<GPU_FB_T, DYN_T>::k(const float * x_act, const float * x_goal,
                                       const float t, float * theta,
                                       float* control_output) {
  float * fb_gain_t = &fb_gains_[DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * (int) t];
  float e = 0;
  for (int i = 0; i < DYN_T::STATE_DIM; i++) {
    e = x_act[i] - x_goal[i];
    for(int j = 0; j < DYN_T::CONTROL_DIM; j++) {
      control_output[j] += fb_gain_t[i * DYN_T::CONTROL_DIM + j] * e;
    }
  }
}

template <class GPU_FB_T, class DYN_T>
void DeviceDDPImpl<GPU_FB_T, DYN_T>::copyToDevice() {
  int fb_size = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * num_timesteps_;
  HANDLE_ERROR(cudaMemcpyAsync(this->feedback_d_->fb_gains_,
                               fb_gains_,
                               sizeof(float) * fb_size,
                               cudaMemcpyHostToDevice,
                               this->stream_));
  HANDLE_ERROR( cudaStreamSynchronize(this->stream_) );
}

template <class DYN_T, int NUM_TIMESTEPS>
DDPFeedback<DYN_T, NUM_TIMESTEPS>::DDPFeedback(DYN_T* model, float dt,
                                               int num_timesteps,
                                               cudaStream_t stream) {
  model_ = model;
  this->dt_ = dt;
  this->num_timesteps_ = std::max(num_timesteps, NUM_TIMESTEPS);
  this->gpu_controller_->freeCudaMem(); // Remove allocated CUDA mem from default constructor
  this->gpu_controller_ = std::make_shared<DeviceDDP<DYN_T>>(this->num_timesteps_, stream);
  this->gpu_controller_->GPUSetup();
}

template <class DYN_T, int NUM_TIMESTEPS>
void DDPFeedback<DYN_T, NUM_TIMESTEPS>::initTrackingController() {
  util::DefaultLogger logger;
  bool verbose = false;
  ddp_model_  = std::make_shared<ModelWrapperDDP<DYN_T>>(model_);
  ddp_solver_ = std::make_shared< DDP<ModelWrapperDDP<DYN_T>>>(this->dt_,
                                                               this->num_timesteps_,
                                                               1,
                                                               &logger,
                                                               verbose);
  // TODO: Can be done by setParams() in feedback base class
  // this->params_.Q = q_mat;
  // this->params_.Q_f = q_f_mat;
  // this->params_.R = r_mat;

  result_ = OptimizerResult<ModelWrapperDDP<DYN_T>>();
  result_.feedback_gain = feedback_gain_trajectory(NUM_TIMESTEPS);
  for(int i = 0; i < NUM_TIMESTEPS; i++) {
    result_.feedback_gain[i] = DYN_T::feedback_matrix::Zero();
  }

  for (int i = 0; i < DYN_T::CONTROL_DIM; i++) {
    control_min_(i) = model_->control_rngs_[i].x;
    control_max_(i) = model_->control_rngs_[i].y;
  }

  run_cost_ = std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(this->params_.Q,
                                                                        this->params_.R,
                                                                        NUM_TIMESTEPS);
  terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(this->params_.Q_f);
}

// template <class DYN_T, int NUM_TIMESTEPS>
// DDPFeedback<DYN_T, NUM_TIMESTEPS>::control_array DDPFeedback<DYN_T, NUM_TIMESTEPS>::k(
//     const Eigen::Ref<state_array>& x_act,
//     const Eigen::Ref<state_array>& x_goal,
//     float t,
//     INTERNAL_STATE_T& fb_state) {
//   // TODO INTERNAL_STATE_T probably won't compile
//   control_array u_output = fb_state.fb_gain_traj_[t] * (x_act - x_goal);
//   return u_output;
// }

template <class DYN_T, int NUM_TIMESTEPS>
void DDPFeedback<DYN_T, NUM_TIMESTEPS>::computeFeedbackGains(
    const Eigen::Ref<const state_array>& init_state,
    const Eigen::Ref<const state_trajectory>& goal_traj,
    const Eigen::Ref<const control_trajectory>& control_traj) {

  run_cost_->setTargets(goal_traj.data(), control_traj.data(),
                        NUM_TIMESTEPS);

  terminal_cost_->xf = run_cost_->traj_target_x_.col(NUM_TIMESTEPS - 1);
  result_ = ddp_solver_->run(init_state, control_traj,
                             *ddp_model_, *run_cost_, *terminal_cost_,
                             control_min_, control_max_);
  this->feedback_state_.fb_gain_traj_ = result_.feedback_gain;

  // Copy Feedback Gains into GPU array
  for (size_t i = 0; i < this->result_.feedback_gain.size(); i++) {
    int i_index = i * DYN_T::STATE_DIM * DYN_T::CONTROL_DIM;
    for (size_t j = 0; j < DYN_T::CONTROL_DIM * DYN_T::STATE_DIM; j++) {
      this->gpu_controller_->fb_gains_[i_index + j] = this->result_.feedback_gain[i].data()[j];
    }
  }
}

// template <class DYN_T, int NUM_TIMESTEPS>
// control_array DDPFeedback<DYN_T, NUM_TIMESTEPS>::interpolateFeedback(
//     state_array& state,
//     state_array& target_nominal_state,
//     double rel_time,
//     INTERNAL_STATE_T& fb_state) {
//   // TODO call the feedback controller version directly
//   int lower_idx = (int) (rel_time / dt_);
//   int upper_idx = lower_idx + 1;
//   double alpha = (rel_time - lower_idx * dt_) / dt_;

//   control_array u_fb = (1 - alpha) * k(state, target_nominal_state, lower_idx)
//       + alpha*k(state, target_nominal_state, upper_idx);

//   return u_fb;
// }