/**
 * Created by jason on 10/30/19.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
**/

#ifndef MPPIGENERIC_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_MPPI_CONTROLLER_CUH

#include "curand.h"
// Double check if these are included in mppi_common.h
#include <chrono>

#include <controllers/controller.cuh>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
class VanillaMPPIController : public Controller<DYN_T, COST_T,
                                                MAX_TIMESTEPS,
                                                NUM_ROLLOUTS,
                                                BDIM_X,
                                                BDIM_Y> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // need control_array = ... so that we can initialize
  // Eigen::Matrix with Eigen::Matrix::Zero();
  using control_array = typename Controller<DYN_T, COST_T,
                            MAX_TIMESTEPS,
                            NUM_ROLLOUTS,
                            BDIM_X,
                            BDIM_Y>::control_array;

  using control_trajectory = typename Controller<DYN_T, COST_T,
                            MAX_TIMESTEPS,
                            NUM_ROLLOUTS,
                            BDIM_X,
                            BDIM_Y>::control_trajectory;

  using state_trajectory = typename Controller<DYN_T, COST_T,
                            MAX_TIMESTEPS,
                            NUM_ROLLOUTS,
                            BDIM_X,
                            BDIM_Y>::state_trajectory;

  using state_array = typename Controller<DYN_T, COST_T,
                            MAX_TIMESTEPS,
                            NUM_ROLLOUTS,
                            BDIM_X,
                            BDIM_Y>::state_array;

  using typename Controller<DYN_T, COST_T,
                            MAX_TIMESTEPS,
                            NUM_ROLLOUTS,
                            BDIM_X,
                            BDIM_Y>::sampled_cost_traj;
  /**
   *
   * Public member functions
   */
  // Constructor
  VanillaMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter,
                        float gamma, int num_timesteps,
                        const control_array& control_variance,
                        const control_trajectory& init_control_traj = control_trajectory::Zero(),
                        cudaStream_t stream= nullptr);
  // Empty Constructor used in inheritance
  VanillaMPPIController() {};

  // Destructor
  ~VanillaMPPIController();


  void updateControlNoiseVariance(const control_array& sigma_u);

  control_array getControlVariance() { return control_variance_;};

  float getBaselineCost() {return baseline_;};

  void computeControl(const state_array& state) override;

  /**
   * returns the current control sequence
   */
  control_trajectory getControlSeq() override { return nominal_control_;};

  /**
   * returns the current state sequence
   */
  state_trajectory getStateSeq() override {return nominal_state_;};

  /**
   * Slide the control sequence back n steps
   */
  void slideControlSequence(int steps) override;

  cudaStream_t stream_;

private:
  int num_iters_;  // Number of optimization iterations

  float gamma_; // Value of the temperature in the softmax.
  float normalizer_; // Variable for the normalizing term from sampling.
  float baseline_; // Baseline cost of the system.
  float dt_;

  control_trajectory nominal_control_ = control_trajectory::Zero();
  state_trajectory nominal_state_ = state_trajectory::Zero();
  sampled_cost_traj trajectory_costs_ = {{0}};

  float* initial_state_d_;
  float* nominal_control_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS
  float* nominal_state_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS
  float* trajectory_costs_d_; // Array of size NUM_ROLLOUTS
  float* control_noise_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS

  void computeNominalStateTrajectory(const state_array& x0);



  void copyNominalControlToDevice();
protected:
  int num_timesteps_;
  curandGenerator_t gen_;


  control_array control_variance_ = control_array::Zero();
  float* control_variance_d_; // Array of size DYN_T::CONTROL_DIM
  // WARNING This method is private because it is only called once in the constructor. Logic is required
  // so that CUDA memory is properly reallocated when the number of timesteps changes.
  void setNumTimesteps(int num_timesteps);

  void createAndSeedCUDARandomNumberGen();

  void setCUDAStream(cudaStream_t stream);

  // Allocate CUDA memory for the controller
  virtual void allocateCUDAMemory();

  // Free CUDA memory for the controller
  virtual void deallocateCUDAMemory();

  void copyControlVarianceToDevice();

};


#if __CUDACC__
#include "mppi_controller.cu"
#endif

#endif //MPPIGENERIC_MPPI_CONTROLLER_CUH
