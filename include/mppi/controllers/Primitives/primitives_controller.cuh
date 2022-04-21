/**
 * Created by david fan on 04/11/22.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
 **/

#ifndef MPPIGENERIC_PRIMITIVES_CONTROLLER_CUH
#define MPPIGENERIC_PRIMITIVES_CONTROLLER_CUH

#include <mppi/controllers/controller.cuh>

#include <vector>

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
class PrimitivesController : public Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>
{
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // need control_array = ... so that we can initialize
  // Eigen::Matrix with Eigen::Matrix::Zero();
  using control_array =
      typename Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::control_array;

  using control_trajectory =
      typename Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::control_trajectory;

  using state_trajectory =
      typename Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::state_trajectory;

  using state_array =
      typename Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::state_array;

  using sampled_cost_traj =
      typename Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::sampled_cost_traj;

  using FEEDBACK_GPU =
      typename Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::TEMPLATED_FEEDBACK_GPU;

  /**
   *
   * Public member functions
   */
  // Constructor
  PrimitivesController(DYN_T* model, COST_T* cost, FB_T* fb_controller, float dt, int max_iter, float lambda,
                        float alpha, const Eigen::Ref<const control_array>& control_std_dev,
                        int num_timesteps = MAX_TIMESTEPS,
                        const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                        cudaStream_t stream = nullptr);

  // Destructor
  ~PrimitivesController();

  std::string getControllerName()
  {
    return "Primitives";
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
  
  void setNumPrimitiveIterations(int new_num_iter)
  {
    num_primitive_iters_ = new_num_iter;
  }

  int getNumPrimitiveIterations()
  {
    return num_primitive_iters_;
  }

  void setColoredNoiseExponents(std::vector<float>& new_exponents)
  {
    colored_noise_exponents_ = new_exponents;
  }

  float getColoredNoiseExponent(int index)
  {
    return colored_noise_exponents_[index];
  }

  void setPiecewiseSegments(int segments)
  {
    num_piecewise_segments_ = segments;
  }

  int getPiecewiseSegments()
  {
    return num_piecewise_segments_;
  }

  void setScalePiecewiseNoise(std::vector<float>& new_scale)
  {
    scale_piecewise_noise_ = new_scale;
  }

  std::vector<float> getScalePiecewiseNoise()
  {
    return scale_piecewise_noise_;
  }

  void setFracRandomNoiseTraj(float frac_random_noise_traj)
  {
    frac_random_noise_traj_ = frac_random_noise_traj;
  }
  float getFracRandomNoiseTraj()
  {
    return frac_random_noise_traj_;
  }

  void setStateLeashLength(float new_state_leash, int index = 0)
  {
    state_leash_dist_[index] = new_state_leash;
  }

  float getStateLeashLength(int index)
  {
    return state_leash_dist_[index];
  }

  void setStoppingCostThreshold(float new_stopping_cost_threshold)
  {
    stopping_cost_threshold_ = new_stopping_cost_threshold;
  }

  float getStoppingCostThreshold()
  {
    return stopping_cost_threshold_;
  }

  void setHysteresisCostThreshold(float new_hysteresis_cost_threshold)
  {
    hysteresis_cost_threshold_ = new_hysteresis_cost_threshold;
  }
  
  float getHysteresisCostThreshold()
  {
    return hysteresis_cost_threshold_;
  }

  void calculateSampledStateTrajectories() override;

protected:
  void computeStateTrajectory(const Eigen::Ref<const state_array>& x0);

  void computeStoppingTrajectory(const Eigen::Ref<const state_array>& x0);
  void smoothControlTrajectory();
  int num_primitive_iters_;
  int num_piecewise_segments_ = 5;
  std::vector<float> scale_piecewise_noise_;
  float frac_random_noise_traj_ = 0.1;
  std::vector<float> colored_noise_exponents_;
  float state_leash_dist_[DYN_T::STATE_DIM] = { 0 };
  int leash_jump_ = 1;
  float stopping_cost_threshold_ = 1.0e8;
  float hysteresis_cost_threshold_ = 0.0;

private:
  // ======== MUST BE OVERWRITTEN =========
  void allocateCUDAMemory();
  // ======== END MUST BE OVERWRITTEN =====
};

#if __CUDACC__
#include "primitives_controller.cu"
#endif

#endif  // MPPIGENERIC_PRIMITIVES_CONTROLLER_CUH
