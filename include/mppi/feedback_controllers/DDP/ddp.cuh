/*
 * Created on Sun Sep 28 2020 by Bogdan
 */

#ifndef FEEDBACK_CONTROLLERS_DDP_CUH_
#define FEEDBACK_CONTROLLERS_DDP_CUH_

#include <mppi/feedback_controllers/feedback.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <mppi/ddp/ddp_tracking_costs.h>
#include <mppi/ddp/ddp.h>
#include <mppi/ddp/util.h>
#include <mppi/utils/math_utils.h>

template <class DYN_T>
struct DDPParams
{
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

  StateCostWeight Q = StateCostWeight::Identity();
  Hessian Q_f = Hessian::Identity();
  ControlCostWeight R = ControlCostWeight::Identity();
  int num_iterations = 1;
};

template <class DYN_T>
struct DDPFeedbackState : GPUState
{
  /**
   * Variables
   **/
  float* fb_gain_traj_ = nullptr;
  float* fb_gain_traj_d_ = nullptr;
  int num_timesteps_ = 0;
  cudaStream_t stream_ = 0;
  DDPFeedbackState(int num_timesteps = 1, cudaStream_t stream = 0);

  DDPFeedbackState(const DDPFeedbackState<DYN_T>& other);  // deep copy constructor

  ~DDPFeedbackState();

  DDPFeedbackState<DYN_T>& operator=(DDPFeedbackState<DYN_T> other);

  template <typename D_T>
  friend void swap(DDPFeedbackState<D_T>& first, DDPFeedbackState<D_T>& second);

  /**
   * Methods
   **/

  __host__ void allocateCUDAMemory();

  __host__ void copyToDevice(bool synchronize = true);

  __host__ void deallocateCUDAMemory();

  __host__ __device__ int getNumTimesteps() const;

  __host__ __device__ float* getFeedbackGainPtr() const;

  __host__ __device__ const float* getConstFeedbackGainPtr() const;

  bool isEqual(const DDPFeedbackState<DYN_T>& other) const;

  void setNumTimesteps(const int num_timesteps);

  void setCUDAStream(const cudaStream_t stream);

  __host__ __device__ std::size_t size() const;
};

/**
 * Needed for Test in base_plant_tester.cu
 **/
template <class DYN_T>
inline bool operator==(const DDPFeedbackState<DYN_T>& lhs, const DDPFeedbackState<DYN_T>& rhs)
{
  return lhs.isEqual(rhs);
};

/**
 * DDP GPU Controller class starting point. This class is where the actual
 * methods for DDP on the GPU are implemented but it is not used directly since
 * setting up the GPU_FB_T value would be painful
 */
template <class GPU_FB_T, class DYN_T>
class DeviceDDPImpl : public GPUFeedbackController<GPU_FB_T, DYN_T, DDPFeedbackState<DYN_T>>
{
public:
  using PARAMS_T = DDPFeedbackState<DYN_T>;
  using PARENT_CLASS = GPUFeedbackController<GPU_FB_T, DYN_T, DDPFeedbackState<DYN_T>>;
  // static const int SHARED_MEM_REQUEST_BLK_BYTES = DYN_T::CONTROL_DIM * DYN_T::STATE_DIM;
  DeviceDDPImpl(int num_timesteps, cudaStream_t stream = 0);
  DeviceDDPImpl(cudaStream_t stream = 0) : PARENT_CLASS(stream)
  {
    this->state_.setNumTimesteps(1);
  };

  void allocateCUDAMemory();
  void deallocateCUDAMemory();
  void copyToDevice(bool synchronize = true);

  void setNumTimesteps(const int num_timesteps);

  __host__ __device__ int getNumTimesteps() const;

  __device__ void k(const float* __restrict__ x_act, const float* __restrict__ x_goal, const int t,
                    float* __restrict__ theta, float* __restrict__ control_output);
};

/**
 * Alias class for DDP GPU Controller. This sets up the class derivation correctly and is
 * used inside of the CPU version of DDP
 */
template <class DYN_T>
class DeviceDDP : public DeviceDDPImpl<DeviceDDP<DYN_T>, DYN_T>
{
public:
  DeviceDDP(int num_timesteps, cudaStream_t stream = 0)
    : DeviceDDPImpl<DeviceDDP<DYN_T>, DYN_T>(num_timesteps, stream){};

  DeviceDDP(cudaStream_t stream = 0) : DeviceDDPImpl<DeviceDDP<DYN_T>, DYN_T>(stream){};
};

/**
 * CPU Class for DDP. This is what the user should interact with
 */
template <class DYN_T>
class DDPFeedback : public FeedbackController<DeviceDDP<DYN_T>, DDPParams<DYN_T>>
{
public:
  /**
   * Aliases
   **/
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;
  typedef FeedbackController<DeviceDDP<DYN_T>, DDPParams<DYN_T>> PARENT_CLASS;

  using control_array = typename PARENT_CLASS::control_array;
  using state_array = typename PARENT_CLASS::state_array;
  using state_trajectory = typename PARENT_CLASS::state_trajectory;
  using control_trajectory = typename PARENT_CLASS::control_trajectory;
  using INTERNAL_STATE_T = typename PARENT_CLASS::TEMPLATED_FEEDBACK_STATE;
  using feedback_gain_matrix = typename DYN_T::feedback_matrix;
  using square_state_matrix = typename DDPParams<DYN_T>::StateCostWeight;
  using square_control_matrix = typename DDPParams<DYN_T>::ControlCostWeight;

  /**
   * Variables
   **/
  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;
  OptimizerResult<ModelWrapperDDP<DYN_T>> result_;

  control_array control_min_;
  control_array control_max_;
  DYN_T* model_;

  DDPFeedback(DYN_T* model, float dt, int num_timesteps = 1, cudaStream_t stream = 0);

  void setParams(const DDPParams<DYN_T>& params) override;

  void initTrackingController();

  control_array k_(const Eigen::Ref<const state_array>& x_act, const Eigen::Ref<const state_array>& x_goal, int t,
                   INTERNAL_STATE_T& fb_state)
  {
    int index = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t;
    Eigen::Map<const feedback_gain_matrix> fb_gain(&(fb_state.getConstFeedbackGainPtr()[index]));
    control_array u_output = fb_gain * (x_act - x_goal);
    return u_output;
  }

  feedback_gain_trajectory getFeedbackGainsEigen()
  {
    return result_.feedback_gain;
  }

  void setNumTimesteps(const int num_timesteps);

  void computeFeedback(const Eigen::Ref<const state_array>& init_state,
                       const Eigen::Ref<const state_trajectory>& goal_traj,
                       const Eigen::Ref<const control_trajectory>& control_traj);
};

#ifdef __CUDACC__
#include "ddp.cu"
#endif

#endif  // FEEDBACK_CONTROLLERS_DDP_CUH_
