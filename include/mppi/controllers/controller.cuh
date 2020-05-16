//
// Created by jason on 10/30/19.
//

#ifndef MPPIGENERIC_CONTROLLER_CUH
#define MPPIGENERIC_CONTROLLER_CUH

#include <array>
#include <Eigen/Core>
#include <chrono>
#include <mppi/ddp/util.h>
#include "curand.h"

#include <mppi/core/mppi_common.cuh>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
         int BDIM_X, int BDIM_Y>
class Controller {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * typedefs for access to templated class from outside classes
   */
  typedef DYN_T TEMPLATED_DYNAMICS;
  typedef COST_T TEMPLATED_COSTS;
  // MAX_TIMESTEPS is defined as an upper bound, if lower that region is just ignored when calculating control
  // does not reallocate cuda memory
  int num_timesteps_ = MAX_TIMESTEPS;

  /**
   * Aliases
   */
   // Control typedefs
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, MAX_TIMESTEPS> control_trajectory; // A control trajectory
//  typedef util::NamedEigenAlignedVector<control_trajectory> sampled_control_traj;
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> K_matrix;

  // State typedefs
  using state_array = typename DYN_T::state_array;
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM, MAX_TIMESTEPS> state_trajectory; // A state trajectory
//  typedef util::NamedEigenAlignedVector<state_trajectory> sampled_state_traj;

  // Cost typedefs
  typedef Eigen::Matrix<float, MAX_TIMESTEPS, 1> cost_trajectory;
  typedef Eigen::Matrix<float, NUM_ROLLOUTS, 1> sampled_cost_traj;
//  typedef std::array<float, MAX_TIMESTEPS> cost_trajectory; // A cost trajectory
//  typedef std::array<float, NUM_ROLLOUTS> sampled_cost_traj; // All costs sampled for all rollouts

  Controller(DYN_T* model, COST_T* cost, float dt, int max_iter, float gamma,
          const Eigen::Ref<const control_array>& control_variance,
          int num_timesteps = MAX_TIMESTEPS,
          const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
          cudaStream_t stream = nullptr) {
    model_ = model;
    cost_ = cost;
    dt_ = dt;
    num_iters_ = max_iter;
    gamma_ = gamma;
    num_timesteps_ = num_timesteps;

    control_variance_ = control_variance;
    control_ = init_control_traj;

    // Create the random number generator
    createAndSeedCUDARandomNumberGen();

    // Bind the model and control to the given stream
    setCUDAStream(stream);

    // Call the GPU setup functions of the model and cost
    this->model_->GPUSetup();
    this->cost_->GPUSetup();

    /**
     * When implementing your own version make sure to write your own allocateCUDAMemroy and call it from the constructor
     * along with any other methods to copy memory to the device and back
     */
  }

  // TODO should be private with test as a friend to ensure it is only used in testing
  Controller() = default;

  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a basePlant reference
   */
  virtual ~Controller() {
    // Free the CUDA memory of every object
    this->model_->freeCudaMem();
    this->cost_->freeCudaMem();

    // Free the CUDA memory of the controller
    deallocateCUDAMemory();
  }

  // =================== METHODS THAT SHOULD HAVE NO DEFAULT ==========================
  // ======== PURE VIRTUAL =========
  /**
   * Given a state, calculates the optimal control sequence using MPPI according
   * to the cost function used as part of the template
   * @param state - the current state from which we would like to calculate
   * a control sequence
   */
  virtual void computeControl(const Eigen::Ref<const state_array>& state) = 0;

  /**
   * Calculate new feedback gains
   */
  virtual void computeFeedbackGains(const Eigen::Ref<const state_array>& state) = 0;

  /**
   * Slide the control sequence back
   */
  virtual void slideControlSequence(int steps) = 0;

  /**
   * Used to update the importance sampler
   * @param nominal_control the new nominal control sequence to sample around
   */
  virtual void updateImportanceSampler(const Eigen::Ref<const control_trajectory>& nominal_control) {
    // TODO copy to device new control sequence
  }

  // ================ END OF MNETHODS WITH NO DEFAULT =============
  // ======== PURE VIRTUAL END =====

  /**
   * returns the current control sequence
   */
  virtual control_trajectory getControlSeq() {
    return control_;
  };

  /**
   * Gets the state sequence of the nominal trajectory
   */
  virtual state_trajectory getStateSeq() {
    return state_;
  }

  /**
   * Return the current minimal cost sequence
   */
  virtual cost_trajectory getCostSeq() {
    // TODO
    return cost_trajectory();
  };

  /**
   * Return all the sampled costs sequences
   */
  virtual sampled_cost_traj getSampledCostSeq() {
    return trajectory_costs_;
  };

  /**
   * Return control feedback gains
   */
 virtual K_matrix getFeedbackGains() {
     K_matrix empty_feedback_gain;
     return empty_feedback_gain;
 };

  control_array getControlVariance() { return control_variance_;};

  float getBaselineCost() {return baseline_;};
  float getNormalizerCost() {return normalizer_;};

  /**
   * return the entire sample of control sequences
   */
//  virtual sampled_control_traj getSampledControlSeq() {
//    return sampled_control_traj();
//  };

  /**
   * Return all the sampled states sequences
   */
//  virtual sampled_state_traj getSampledStateSeq() {
//    return sampled_state_traj();
//  };


  /**
   * Reset Controls
   */
  virtual void resetControls() {};

  void setNumTimesteps(int num_timesteps) {
    // TODO fix the tracking controller as well
    if ((num_timesteps <= MAX_TIMESTEPS) && (num_timesteps > 0)) {
      num_timesteps_ = num_timesteps;
    } else {
      num_timesteps_ = MAX_TIMESTEPS;
      printf("You must give a number of timesteps between [0, %d]\n", MAX_TIMESTEPS);
    }
  }

  /**
   * updates the scaling factor of noise for sampling around the nominal trajectory
   */
  void updateControlNoiseVariance(const Eigen::Ref<const control_array>& sigma_u) {
    //std::cout << control_variance_ << std::endl;
    control_variance_ = sigma_u;
    //std::cout << control_variance_ << std::endl;
    copyControlVarianceToDevice();
  }

  /**
   * Public data members
   */
  DYN_T* model_;
  COST_T* cost_;
  cudaStream_t stream_;

protected:
  // no default protected members
  void deallocateCUDAMemory() {
    cudaFree(control_d_);
    cudaFree(state_d_);
    cudaFree(trajectory_costs_d_);
    cudaFree(control_variance_d_);
    cudaFree(control_noise_d_);
  };
  // TODO smoothControlTrajectory


  int num_iters_;  // Number of optimization iterations
  float dt_;
  float gamma_; // Value of the temperature in the softmax.

  float normalizer_; // Variable for the normalizing term from sampling.
  float baseline_; // Baseline cost of the system.

  curandGenerator_t gen_;
  control_array control_variance_ = control_array::Zero();
  float* control_variance_d_; // Array of size DYN_T::CONTROL_DIM
  float* initial_state_d_; // Array of sizae DYN_T::STATE_DIM * (2 if there is a nominal state)

  // Control history
  Eigen::Matrix<float, 2, DYN_T::CONTROL_DIM> control_history_ = Eigen::Matrix<float, 2, DYN_T::CONTROL_DIM>::Zero();

  // one array of this size is allocated for each state we care about,
  // so it can be the size*N for N nominal states
  // [actual, nominal]
  float* control_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*N
  float* state_d_; // Array of size DYN_T::STATE_DIM*NUM_ROLLOUTS*N
  float* trajectory_costs_d_; // Array of size NUM_ROLLOUTS*N
  float* control_noise_d_; // Array of size DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS*N
  control_trajectory control_ = control_trajectory::Zero();
  state_trajectory state_ = state_trajectory::Zero();
  sampled_cost_traj trajectory_costs_ = sampled_cost_traj::Zero();

  void copyControlVarianceToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(control_variance_d_, control_variance_.data(), sizeof(float)*control_variance_.size(), cudaMemcpyHostToDevice, stream_));
    cudaStreamSynchronize(stream_);
  }

  void copyNominalControlToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(control_d_, control_.data(), sizeof(float)*control_.size(), cudaMemcpyHostToDevice, stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }

  void setCUDAStream(cudaStream_t stream) {
    stream_ = stream;
    this->model_->bindToStream(stream);
    this->cost_->bindToStream(stream);
    curandSetStream(gen_, stream); // requires the generator to be created!
  }

  void createAndSeedCUDARandomNumberGen() {
    // Seed the PseudoRandomGenerator with the CPU time.
    curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    curandSetPseudoRandomGeneratorSeed(gen_, seed);
  }

  /**
   * Allocates CUDA memory for actual states and nominal states if needed
   * @param nominal_size if only actual this should be 0
   */
  void allocateCUDAMemoryHelper(int nominal_size = 0, bool allocate_double_noise = true) {
    if(nominal_size < 0) {
      nominal_size = 1;
      // TODO throw exception
    } else {
      // increment by 1 since actual is not included
      ++nominal_size;
    }
    HANDLE_ERROR(cudaMalloc((void**)&this->initial_state_d_,
                            sizeof(float)*DYN_T::STATE_DIM*nominal_size));
    HANDLE_ERROR(cudaMalloc((void**)&this->control_d_,
                            sizeof(float)*DYN_T::CONTROL_DIM*MAX_TIMESTEPS*nominal_size));
    HANDLE_ERROR(cudaMalloc((void**)&this->state_d_,
                            sizeof(float)*DYN_T::STATE_DIM*MAX_TIMESTEPS*nominal_size));
    HANDLE_ERROR(cudaMalloc((void**)&this->trajectory_costs_d_,
                            sizeof(float)*NUM_ROLLOUTS*nominal_size));
    HANDLE_ERROR(cudaMalloc((void**)&this->control_variance_d_,
                            sizeof(float)*DYN_T::CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&this->control_noise_d_,
                            sizeof(float)*DYN_T::CONTROL_DIM*MAX_TIMESTEPS*NUM_ROLLOUTS* (allocate_double_noise ? nominal_size : 1)));
  }

  // TODO all the copy to device functions to streamline process
private:
  // ======== MUST BE OVERWRITTEN =========
  void allocateCUDAMemory() {
    allocateCUDAMemoryHelper();
  };
  /**
   * TODO all copy to device and back functions implemented for specific controller
   * When you write your own you must control when synchronize stream is called
   */
  // ======== END MUST BE OVERWRITTEN =====
};

#endif //MPPIGENERIC_CONTROLLER_CUH
