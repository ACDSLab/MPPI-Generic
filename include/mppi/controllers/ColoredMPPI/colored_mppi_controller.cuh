/**
 * Created by jason on 10/30/19.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
 **/

#ifndef MPPIGENERIC_COLORED_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_COLORED_MPPI_CONTROLLER_CUH

#include <mppi/controllers/controller.cuh>

#include <vector>

template <int C_DIM, int MAX_TIMESTEPS>
struct ColoredMPPIParams : ControllerParams<C_DIM, MAX_TIMESTEPS>
{
  std::vector<float> colored_noise_exponents_;
};

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y,
          class PARAMS_T = ColoredMPPIParams<DYN_T::CONTROL_DIM, MAX_TIMESTEPS>>
class ColoredMPPIController
  : public Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y, PARAMS_T>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y, PARAMS_T> PARENT_CLASS;
  // need control_array = ... so that we can initialize
  // Eigen::Matrix with Eigen::Matrix::Zero();
  using control_array = typename PARENT_CLASS::control_array;

  using control_trajectory = typename PARENT_CLASS::control_trajectory;

  using state_trajectory = typename PARENT_CLASS::state_trajectory;

  using state_array = typename PARENT_CLASS::state_array;

  using sampled_cost_traj = typename PARENT_CLASS::sampled_cost_traj;

  using FEEDBACK_GPU = typename PARENT_CLASS::TEMPLATED_FEEDBACK_GPU;

  /**
   *
   * Public member functions
   */
  // Constructor
  ColoredMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, float dt, int max_iter, float lambda,
                        float alpha, const Eigen::Ref<const control_array>& control_std_dev,
                        int num_timesteps = MAX_TIMESTEPS,
                        const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                        cudaStream_t stream = nullptr);

  ColoredMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, PARAMS_T& params,
                        cudaStream_t stream = nullptr);

  // Destructor
  ~ColoredMPPIController();

  std::string getControllerName()
  {
    return "Colored MPPI";
  };

  /**
   * computes a new control sequence
   * @param state starting position
   */
  void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride = 1) override;

  /**
   * Slide the control sequence back n steps
   */
  void slideControlSequence(int optimization_stride) override;

  void setPercentageSampledControlTrajectories(float new_perc)
  {
    this->setPercentageSampledControlTrajectoriesHelper(new_perc, 1);
  }

  void setColoredNoiseExponents(std::vector<float>& new_exponents)
  {
    this->params_.colored_noise_exponents_ = new_exponents;
  }

  float getColoredNoiseExponent(int index)
  {
    return this->params_.colored_noise_exponents_[index];
  }

  void setStateLeashLength(float new_state_leash, int index = 0)
  {
    state_leash_dist_[index] = new_state_leash;
  }

  float getStateLeashLength(int index)
  {
    return state_leash_dist_[index];
  }

  void calculateSampledStateTrajectories() override;

protected:
  void computeStateTrajectory(const Eigen::Ref<const state_array>& x0);

  void smoothControlTrajectory();
  float state_leash_dist_[DYN_T::STATE_DIM] = { 0 };
  int leash_jump_ = 1;

private:
  // ======== MUST BE OVERWRITTEN =========
  void allocateCUDAMemory();
  // ======== END MUST BE OVERWRITTEN =====
};

#if __CUDACC__
#include "colored_mppi_controller.cu"
#endif

#endif  // MPPIGENERIC_COLORED_MPPI_CONTROLLER_CUH
