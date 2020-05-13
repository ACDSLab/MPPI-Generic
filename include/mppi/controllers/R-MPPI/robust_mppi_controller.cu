#include "robust_mppi_controller.cuh"
#include <exception>

#define RobustMPPI RobustMPPIController<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
RobustMPPI::~RobustMPPIController() {
  deallocateNominalStateCandidateMemory();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::getInitNominalStateCandidates(
        const Eigen::Ref<const state_array>& nominal_x_k,
        const Eigen::Ref<const state_array>& nominal_x_kp1,
        const Eigen::Ref<const state_array>& real_x_kp1) {

  Eigen::MatrixXf points(DYN_T::STATE_DIM, 3);
  points << nominal_x_k, nominal_x_kp1 , real_x_kp1;
  auto candidates = points*line_search_weights;
  for (int i = 0; i < num_candidate_nominal_states; ++i) {
    candidate_nominal_states[i] = candidates.col(i);
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::resetCandidateCudaMem() {
  deallocateNominalStateCandidateMemory();
  HANDLE_ERROR(cudaMalloc((void**)&importance_sampling_states_d_,
          sizeof(float)*DYN_T::STATE_DIM*num_candidate_nominal_states));
  HANDLE_ERROR(cudaMalloc((void**)&importance_sampling_costs_d_,
                          sizeof(float)*num_candidate_nominal_states));
  HANDLE_ERROR(cudaMalloc((void**)&importance_sampling_strides_d_,
                          sizeof(float)*num_candidate_nominal_states));

  // Set flag so that the we know cudamemory is allocated
  importance_sampling_cuda_mem_init = true;
}



template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::deallocateNominalStateCandidateMemory() {
  if (importance_sampling_cuda_mem_init) {
    HANDLE_ERROR(cudaFree(importance_sampling_states_d_));
    HANDLE_ERROR(cudaFree(importance_sampling_costs_d_));
    HANDLE_ERROR(cudaFree(importance_sampling_strides_d_));

    // Set flag so that we know cudamemory has been freed
    importance_sampling_cuda_mem_init = false;
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::updateNumCandidates(int new_num_candidates) {

  // New number must be odd and greater than 3
  if (new_num_candidates < 3) {
    std::cerr << "ERROR: number of candidates must be greater or equal to 3\n";
    std::terminate();
  }
  if (new_num_candidates % 2 == 0) {
    std::cerr << "ERROR: number of candidates must be odd\n";
    std::terminate();
  }
  // Set the new value of the number of candidates
  num_candidate_nominal_states = new_num_candidates;

  // Resize the vector holding the candidate nominal states
  candidate_nominal_states.resize(num_candidate_nominal_states);

  // Resize the matrix holding the importance sampler strides
  importance_sampler_strides.resize(1, num_candidate_nominal_states);

  // Deallocate and reallocate cuda memory
  resetCandidateCudaMem();

  // Recompute the line search weights based on the number of candidates
  computeLineSearchWeights();
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::computeLineSearchWeights() {
  line_search_weights.resize(3, num_candidate_nominal_states);

  // For a given setup, this never changes.... why recompute every time?
  int num_candid_over_2 = num_candidate_nominal_states/2;
  for (int i = 0; i < num_candid_over_2 + 1; i++){
    line_search_weights(0, i) = 1 - i/float(num_candid_over_2);
    line_search_weights(1, i) = i/float(num_candid_over_2);
    line_search_weights(2, i) = 0.0;
  }
  for (int i = 1; i < num_candid_over_2 + 1; i++){
    line_search_weights(0, num_candid_over_2 + i) = 0.0;
    line_search_weights(1, num_candid_over_2 + i) = 1 - i/float(num_candid_over_2);
    line_search_weights(2, num_candid_over_2 + i) = i/float(num_candid_over_2);
  }
}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::computeImportanceSamplerStride(int stride) {
  Eigen::MatrixXf stride_vec(1,3);
  stride_vec << 0, stride, stride;

  // Perform matrix multiplication, convert to array so that we can round the floats to the nearest
  // integer. Then cast the resultant float array to an int array. Then set equal to our int matrix.
  importance_sampler_strides = (stride_vec*line_search_weights).array().round().template cast<int>();

}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::setCudaStream(cudaStream_t stream) {

}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::allocateCUDAMemory() {
// Allocate memory for the control noise
  HANDLE_ERROR( cudaMalloc((void**)&control_noise_d_, 2*NUM_ROLLOUTS*MAX_TIMESTEPS*CONTROL_DIM*sizeof(float)));

}

template<class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void RobustMPPI::deallocateCudaMemory() {

}

/******************************************************************************
//MPPI Kernel Implementations and helper launch files
*******************************************************************************/
//#define BLOCKSIZE_X RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::BLOCKSIZE_X
//#define BLOCKSIZE_Y RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::BLOCKSIZE_Y
//#define BLOCKSIZE_Z 2 //Z blocksize of 2 for the two copies of the state (nominal and actual)
//#define BLOCKSIZE_WRX RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::BLOCKSIZE_WRX
//#define STATE_DIM DYNAMICS_T::STATE_DIM
//#define CONTROL_DIM DYNAMICS_T::CONTROL_DIM
//#define SHARED_MEM_REQUEST_GRD DYNAMICS_T::SHARED_MEM_REQUEST_GRD
//#define SHARED_MEM_REQUEST_BLK DYNAMICS_T::SHARED_MEM_REQUEST_BLK
//#define NUM_ROLLOUTS RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::NUM_ROLLOUTS
//
//template<class DYN_T, class COST_T, int BLOCKSIZE_X, int BLOCKSIZE_Y,
//         int NUM_ROLLOUTS, int BLOCKSIZE_Z>
//  __global__ void rolloutKernel(DYN_T* dynamics, COST_T* costs,
//                               float dt,
//                               int num_timesteps,
//                               float* x_d,
//                               float* u_d,
//                               float* du_d,
//                               float* sigma_u_d,
//                               float* trajectory_costs_d,
//                               float* feedback_gains_d) {
//    //Get thread and block id
//    int thread_idx = threadIdx.x;
//    int thread_idy = threadIdx.y;
//    int thread_idz = threadIdx.z;
//    int block_idx = blockIdx.x;
//    int global_idx = BLOCKSIZE_X * block_idx + thread_idx;
//
//    //Create shared state and control arrays
//    __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
//    __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
//    __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
//    __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
//    __shared__ float sigma_u[DYN_T::CONTROL_DIM];
//    __shared__ float fb_u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM];
//
//  //Create a shared array for the dynamics model to use
//  __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X];
//
//    //Create local state, state dot and controls
//    float* x;
//    float* xdot;
//    float* u;
//    float* du;
//    float* feedback_u;
//    // float* sigma_u;
//
//    //Initialize running cost and total cost
//    float running_cost = 0;
//    //Load global array to shared array
//    if (global_idx < NUM_ROLLOUTS) {
//      x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
//      xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
//      u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
//      du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
//      feedback_u = &fb_u_shared[thread_idx * DYN_T::CONTROL_DIM];
//
//      // sigma_u = &sigma_u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
//    }
//    __syncthreads();
//    loadGlobalToShared(DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, NUM_ROLLOUTS,
//                       BLOCKSIZE_Y, global_idx, thread_idy,
//                       thread_idz, x_d, sigma_u_d, x, xdot, u, du, sigma_u);
//    __syncthreads();
//
//
//  if (global_idx < NUM_ROLLOUTS) {
//    /*<----Start of simulation loop-----> */
//    for (int t = 0; t < num_timesteps; t++) {
//      //Load noise trajectories scaled by the exploration factor
//      injectControlNoise(DYN_T::CONTROL_DIM, BLOCKSIZE_Y, NUM_ROLLOUTS, num_timesteps,
//                         t, global_idx, thread_idy, u_d, du_d, sigma_u, u, du);
//      __syncthreads();
//
//      if (thread_idz == 0){ // Only calculate in the actual dynamics
//        // TODO: Replace for loops of int m with parallelization over threads
//        for (int m = thread_idy; m < DYN_T::CONTROL_DIM; m += blockDim.y) {
//          feedback_u[m] = 0;
//        }
//        float e;
//        for (int k = 0; k < DYN_T::STATE_DIM; k++){
//          e = x[k] - x[blockDim.x * DYN::STATE_DIM + k];
//          // e = (state_shared[tdx*DYN_T::STATE_DIM + k] - state_shared[blockDim.x*DYN_T::STATE_DIM + tdx*DYN_T::STATE_DIM + k]);
//          for (int m = thread_idy; m < DYN_T::CONTROL_DIM; m += blockDim.y) {
//            feedback_u[m] += feedback_gains_d[t * DYN_T::STATE_DIM * DYN_T::CONTROL_DIM + DYN_T::STATE_DIM * m + k] * e;
//          }
//          // delta_steering += feedback_gains_d[t*DYN_T::STATE_DIM*DYN_T::CONTROL_DIM + k]*e;
//          // delta_throttle += feedback_gains_d[t*DYN_T::STATE_DIM*DYN_T::CONTROL_DIM + DYN_T::STATE_DIM + k]*e;
//        }
//        for (int m = thread_idy; m < DYN_T::CONTROL_DIM; m += blockDim.y) {
//          u[m] += feedback_u[m];
//        }
//        // u[0] += delta_steering;
//        // u[1] += delta_throttle;
//      }
//      __syncthreads();
//
//      // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
//      // usually just control clamping
//      dynamics->enforceConstraints(x, &du_d[global_idx*num_timesteps*DYN_T::CONTROL_DIM + t*DYN_T::CONTROL_DIM]);
//      dynamics->enforceConstraints(x, u);
//
//    __syncthreads();
//
//      //Accumulate running cost
//      running_cost += costs->computeRunningCost(x, u, du, sigma_u, t)*dt;
//      __syncthreads();
//
//      //Compute state derivatives
//      dynamics->computeStateDeriv(x, u, xdot, theta_s);
//      __syncthreads();
//
//      //Increment states
//      dynamics->updateState(x, xdot, dt);
//      __syncthreads();
//    }
//    //Compute terminal cost and the final cost for each thread
//    computeAndSaveCost(NUM_ROLLOUTS, global_idx, costs, x, running_cost,
//                       trajectory_costs_d + thread_idz * NUM_ROLLOUTS);
//  }
//
//    __syncthreads();
//  }
//
//// template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//// __global__ void rolloutKernel(int num_timesteps, float* state_d, float* feedback_gains_d, float* U_d, float* du_d, float* nu_d,
////                               float* augmented_nominal_costs_d, float* augmented_real_costs_d, float* pure_real_costs_d,
////                               DYNAMICS_T dynamics_model, COSTS_T mppi_costs, int lambda, int opt_delay)
//// {
////   int i,j;
////   int tdx = threadIdx.x;
////   int tdy = threadIdx.y;
////   int tdz = threadIdx.z;
////   int bdx = blockIdx.x;
//
////   //Initialize the local state, controls, and noise
////   float* s;
////   float* s_der;
////   float* u;
////   float* du;
////   int* crash;
//
////   //Create shared arrays for holding state and control data.
////   __shared__ float state_shared[BLOCKSIZE_X*BLOCKSIZE_Z*STATE_DIM];
////   __shared__ float state_der_shared[BLOCKSIZE_X*BLOCKSIZE_Z*STATE_DIM];
////   __shared__ float control_shared[BLOCKSIZE_X*BLOCKSIZE_Z*CONTROL_DIM];
////   __shared__ float control_var_shared[BLOCKSIZE_X*BLOCKSIZE_Z*CONTROL_DIM];
////   __shared__ int crash_status[BLOCKSIZE_Z*BLOCKSIZE_X];
////   //Create a shared array for the dynamics model to use
////   __shared__ float theta[SHARED_MEM_REQUEST_GRD + SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X*BLOCKSIZE_Z];
//
////   //Every thread uses the same shared exploration variance
////   __shared__ float nu[CONTROL_DIM];
//
////   //Initialize trajectory cost
////   float running_cost = 0;
////   float pure_real_unbiased_cost = 0;
//
////   //Initialize the dynamics model.
////   dynamics_model.cudaInit(theta);
//
////   int global_idx = BLOCKSIZE_X*bdx + tdx;
//
////   //Initialize shared and local variables
////   s = &state_shared[blockDim.x*STATE_DIM*tdz + tdx*STATE_DIM];
////   s_der = &state_der_shared[blockDim.x*STATE_DIM*tdz + tdx*STATE_DIM];
////   u = &control_shared[blockDim.x*CONTROL_DIM*tdz + tdx*CONTROL_DIM];
////   du = &control_var_shared[blockDim.x*CONTROL_DIM*tdz + tdx*CONTROL_DIM];
////   crash = &crash_status[tdz*blockDim.x + tdx];
//
////   //Load the initial state, nu, and zero the noise
////   for (i = tdy; i < STATE_DIM; i+= blockDim.y) {
////     s[i] = state_d[STATE_DIM*tdz + i];
////     s_der[i] = 0;
////   }
////   //Load nu
////   for (i = tdy; i < CONTROL_DIM; i+= blockDim.y) {
////     u[i] = 0;
////     du[i] = 0;
////   }
////   for (i = blockDim.y*blockDim.x*tdz + blockDim.x*tdy + tdx; i < CONTROL_DIM; i+= blockDim.z*blockDim.x*blockDim.y){
////     nu[i] = nu_d[i];
////   }
//
////   float delta_steering = 0;
////   float delta_throttle = 0;
////   crash[0] = 0;
////   __syncthreads();
//
////   /*<----Start of simulation loop-----> */
////   for (i = 0; i < num_timesteps; i++) {
//
////     //Sample the control to apply
////     for (j = tdy; j < CONTROL_DIM; j+= blockDim.y) {
////       //Noise free rollout
////       if (global_idx == 0 || i < opt_delay) { //Don't optimize variables that are already being executed
////         du[j] = 0.0;
////         u[j] = U_d[i*CONTROL_DIM + j];
////       }
////       else {
////         du[j] = du_d[(tdz*NUM_ROLLOUTS*num_timesteps*CONTROL_DIM) + CONTROL_DIM*num_timesteps*(BLOCKSIZE_X*bdx + tdx) + i*CONTROL_DIM + j]*nu[j];
////         u[j] = U_d[i*CONTROL_DIM + j] + du[j];
////       }
////       du_d[(tdz*NUM_ROLLOUTS*num_timesteps*CONTROL_DIM) + CONTROL_DIM*num_timesteps*(BLOCKSIZE_X*bdx + tdx) + i*CONTROL_DIM + j] = u[j];
////     }
////     __syncthreads();
////     if (tdy == 0){
////       delta_steering = 0;
////       delta_throttle = 0;
////       float e;
////       for (int k = 0; k < STATE_DIM; k++){
////         e = (state_shared[tdx*STATE_DIM + k] - state_shared[blockDim.x*STATE_DIM + tdx*STATE_DIM + k]);
////         delta_steering += feedback_gains_d[i*STATE_DIM*CONTROL_DIM + k]*e;
////         delta_throttle += feedback_gains_d[i*STATE_DIM*CONTROL_DIM + STATE_DIM + k]*e;
////       }
////       if (tdz == 0){
////         u[0] += delta_steering;
////         u[1] += delta_throttle;
////       }
////     }
////     __syncthreads();
////     //Enforce control and state constraints
////     if (tdy == 0){
////        dynamics_model.enforceConstraints(s, u);
////     }
////     __syncthreads();
////     //Compute the cost of the being in the current state
////     if (tdy == 0 && i > 0 && crash[0] != -1) {
////       //Running average formula
////       float curr_cost = mppi_costs.computeCost(s, u, du, nu, crash, i);
////       if (tdz == 0){ //Compute the control cost with the base distribution as the nominal control
////         curr_cost += lambda*(delta_steering*delta_steering/(nu[0]*nu[0]) + delta_throttle*delta_throttle/(nu[1]*nu[1]));
////         float unbiasing_term = lambda*(delta_steering*du[0]/(nu[0]*nu[0]) + delta_throttle*du[1]/(nu[1]*nu[1]));
////         pure_real_unbiased_cost += (curr_cost + unbiasing_term - pure_real_unbiased_cost)/(1.0*i);
////       }
////       running_cost += (curr_cost - running_cost)/(1.0*i);
////     }
////     //Compute the dynamics
////     dynamics_model.computeStateDeriv(s, u, s_der, theta);
////     __syncthreads();
////     //Update the state
////     dynamics_model.incrementState(s, s_der);
////   }
////   /* <------- End of the simulation loop ----------> */
////   //Write cost results back to global memory.
////   if (global_idx < NUM_ROLLOUTS && tdy == 0 && tdz == 0) {
////     augmented_real_costs_d[BLOCKSIZE_X*bdx + tdx] = running_cost;
////     pure_real_costs_d[BLOCKSIZE_X*bdx + tdx] = pure_real_unbiased_cost;
////   }
////   if (global_idx < NUM_ROLLOUTS && tdy == 0 && tdz == 1) {
////     augmented_nominal_costs_d[BLOCKSIZE_X*bdx + tdx] = running_cost;
////   }
//// }
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//__global__ void initEvalKernel(int num_timesteps, float* states_d, int* strides_d, float* U_d, float* du_d, float* nu_d,
//                              float* costs_d, int samples_per_condition, DYNAMICS_T dynamics_model, COSTS_T mppi_costs, int opt_delay)
//{
//  int i,j;
//  int tdx = threadIdx.x;
//  int tdy = threadIdx.y;
//  int bdx = blockIdx.x;
//
//  //Initialize the local state, controls, and noise
//  float* s;
//  float* s_der;
//  float* u;
//  float* nu;
//  float* du;
//  int* crash;
//
//  //Create shared arrays for holding state and control data.
//  __shared__ float state_shared[BLOCKSIZE_X*STATE_DIM];
//  __shared__ float state_der_shared[BLOCKSIZE_X*STATE_DIM];
//  __shared__ float control_shared[BLOCKSIZE_X*CONTROL_DIM];
//  __shared__ float control_var_shared[BLOCKSIZE_X*CONTROL_DIM];
//  __shared__ float exploration_variance[BLOCKSIZE_X*CONTROL_DIM];
//  __shared__ int crash_status[BLOCKSIZE_X];
//  //Create a shared array for the dynamics model to use
//  __shared__ float theta[SHARED_MEM_REQUEST_GRD + SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X];
//
//  //Initialize trajectory cost
//  float running_cost = 0;
//
//  //Initialize the dynamics model.
//  dynamics_model.cudaInit(theta);
//
//  int global_idx = BLOCKSIZE_X*bdx + tdx;
//  int condition_idx = global_idx / samples_per_condition;
//  int stride = strides_d[condition_idx];
//  //Portion of the shared array belonging to each x-thread index.
//  s = &state_shared[tdx*STATE_DIM];
//  s_der = &state_der_shared[tdx*STATE_DIM];
//  u = &control_shared[tdx*CONTROL_DIM];
//  du = &control_var_shared[tdx*CONTROL_DIM];
//  nu = &exploration_variance[tdx*CONTROL_DIM];
//  crash = &crash_status[tdx];
//  //Load the initial state, nu, and zero the noise
//  for (i = tdy; i < STATE_DIM; i+= blockDim.y) {
//    s[i] = states_d[STATE_DIM*condition_idx + i];
//    s_der[i] = 0;
//  }
//  //Load nu
//  for (i = tdy; i < CONTROL_DIM; i+= blockDim.y) {
//    u[i] = 0;
//    du[i] = 0;
//    nu[i] = nu_d[i];
//  }
//  crash[0] = 0;
//  __syncthreads();
//  /*<----Start of simulation loop-----> */
//  for (i = 0; i < num_timesteps; i++) {
//    for (j = tdy; j < CONTROL_DIM; j+= blockDim.y) {
//    //Noise free rollout
//      if (global_idx % samples_per_condition == 0 || i < opt_delay) { //Don't optimize variables that are already being executed
//        du[j] = 0.0;
//        if (i+stride >= num_timesteps){
//          u[j] = 0;
//        }
//        else{
//          u[j] = U_d[(i+stride)*CONTROL_DIM + j];
//        }
//      }
//      else {
//        du[j] = du_d[CONTROL_DIM*num_timesteps*(BLOCKSIZE_X*bdx + tdx) + i*CONTROL_DIM + j]*nu[j];
//        if (i+stride >= num_timesteps){
//          u[j] = du[j];
//        }
//        else{
//          u[j] = U_d[(i+stride)*CONTROL_DIM + j] + du[j];
//        }
//      }
//    }
//    __syncthreads();
//    if (tdy == 0){
//      dynamics_model.enforceConstraints(s, u);
//    }
//    __syncthreads();
//    //Compute the cost of the being in the current state
//    if (tdy == 0 && i > 0 && crash[0] > -1) {
//      //Running average formula
//      running_cost += (mppi_costs.computeCost(s, u, du, nu, crash, i) - running_cost)/(1.0*i);
//    }
//    //Compute the dynamics
//    dynamics_model.computeStateDeriv(s, u, s_der, theta);
//    __syncthreads();
//    //Update the state
//    dynamics_model.incrementState(s, s_der);
//  }
//  /* <------- End of the simulation loop ----------> */
//  if (tdy == 0) {   //Write cost results back to global memory.
//    costs_d[(BLOCKSIZE_X)*bdx + tdx] = running_cost;// + mppi_costs.terminalCost(s, crash);
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//__global__ void normExpKernel(float* state_costs_d, float gamma, float baseline)
//{
//  int tdx = threadIdx.x;
//  int bdx = blockIdx.x;
//  if (BLOCKSIZE_X*bdx + tdx < NUM_ROLLOUTS) {
//    float cost2go = 0;
//    cost2go = state_costs_d[BLOCKSIZE_X*bdx + tdx] - baseline;
//    state_costs_d[BLOCKSIZE_X*bdx + tdx] = exp(-gamma*cost2go);
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//__global__ void weightedReductionKernel(float* states_d, float* du_d, float* nu_d,
//                                        float normalizer, int num_timesteps)
//{
//  int tdx = threadIdx.x;
//  int bdx = blockIdx.x;
//
//  __shared__ float u_system[STATE_DIM*((NUM_ROLLOUTS-1)/BLOCKSIZE_WRX + 1)];
//  int stride = BLOCKSIZE_WRX;
//
//  float u[CONTROL_DIM];
//
//  int i,j;
//  for (i = 0; i < CONTROL_DIM; i++) {
//    u[i] = 0;
//  }
//
//  for (j = 0; j < CONTROL_DIM; j++) {
//    u_system[tdx*CONTROL_DIM + j] = 0;
//  }
//  __syncthreads();
//
//  if (BLOCKSIZE_WRX*tdx < NUM_ROLLOUTS) {
//    float weight = 0;
//    for (i = 0; i < stride; i++) {
//      if (stride*tdx + i < NUM_ROLLOUTS) {
//        weight = states_d[stride*tdx + i]/normalizer;
//        for (j = 0; j < CONTROL_DIM; j++) {
//          u[j] = du_d[(stride*tdx + i)*(num_timesteps*CONTROL_DIM) + bdx*CONTROL_DIM + j];
//          u_system[tdx*CONTROL_DIM + j] += weight*u[j];
//        }
//      }
//    }
//  }
//  __syncthreads();
//  if (tdx == 0 && bdx < num_timesteps) {
//    for (i = 0; i < CONTROL_DIM; i++) {
//      u[i] = 0;
//    }
//    for (i = 0; i < (NUM_ROLLOUTS-1)/BLOCKSIZE_WRX + 1; i++) {
//      for (j = 0; j < CONTROL_DIM; j++) {
//        u[j] += u_system[CONTROL_DIM*i + j];
//      }
//    }
//    for (i = 0; i < CONTROL_DIM; i++) {
//      du_d[CONTROL_DIM*bdx + i] = u[i];
//    }
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//__global__ void solutionCostKernel(COSTS_T* mppi_costs, float* traj_d, float* traj_costs_d, int num_timesteps)
//{
//  int tdx = threadIdx.x;
//  int bdx = blockIdx.x;
//  int global_idx = bdx*blockDim.x + tdx;
//  if (global_idx < num_timesteps){
//      float u[2];
//      float du[2];
//      float nu[2];
//      int crash = 0;
//      for (int i = 0; i < CONTROL_DIM; i++){
//        u[i] = 0;
//        du[i] = 0;
//        nu[i] = 1.0;
//      }
//      traj_costs_d[tdx] = mppi_costs->computeCost(&traj_d[tdx*STATE_DIM], u, du, nu, &crash, tdx);
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void launchRolloutKernel(int num_timesteps, float* state_d, float* feedback_gains_d, float* U_d, float* du_d, float* nu_d,
//                         float* augmented_nominal_costs_d, float* augmented_real_costs_d, float* pure_real_costs_d,
//                         DYNAMICS_T *dynamics_model, COSTS_T *mppi_costs, int lambda, int opt_delay, cudaStream_t stream)
//{
//  const int GRIDSIZE_X = (NUM_ROLLOUTS-1)/BLOCKSIZE_X + 1;
//  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 2);
//  dim3 dimGrid(GRIDSIZE_X, 1, 1);
//  rolloutKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y><<<dimGrid, dimBlock, 0, stream>>>(num_timesteps, state_d, feedback_gains_d, U_d,
//                                                                du_d, nu_d, augmented_nominal_costs_d, augmented_real_costs_d,
//                                                                pure_real_costs_d, *dynamics_model, *mppi_costs, lambda, opt_delay);
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void launchInitEvalKernel(int num_timesteps, float* states_d, int* strides_d, float* U_d, float* du_d, float* nu_d,
//                         float* costs_d, int samples_per_condition, DYNAMICS_T *dynamics_model, COSTS_T *mppi_costs,
//                         int opt_delay, cudaStream_t stream)
//{
//  int GRIDSIZE_X = 9*samples_per_condition/BLOCKSIZE_X;
//  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
//  dim3 dimGrid(GRIDSIZE_X, 1, 1);
//  initEvalKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y><<<dimGrid, dimBlock, 0, stream>>>(num_timesteps, states_d, strides_d, U_d,
//    du_d, nu_d, costs_d, samples_per_condition, *dynamics_model, *mppi_costs, opt_delay);
//}
//
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void launchNormExpKernel(float* costs_d, float gamma, float baseline, cudaStream_t stream)
//{
//  dim3 dimBlock(BLOCKSIZE_X, 1, 1);
//  dim3 dimGrid((NUM_ROLLOUTS-1)/BLOCKSIZE_X + 1, 1, 1);
//  normExpKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y><<<dimGrid, dimBlock, 0, stream>>>(costs_d, gamma, baseline);
//}
//
////Launches the multiplication and reduction kernel
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void launchWeightedReductionKernel(float* state_costs_d, float* du_d, float* nu_d,
//                                  float normalizer, int num_timesteps, cudaStream_t stream)
//{
//    dim3 dimBlock((NUM_ROLLOUTS-1)/BLOCKSIZE_WRX + 1, 1, 1);
//    dim3 dimGrid(num_timesteps, 1, 1);
//    weightedReductionKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y><<<dimGrid, dimBlock, 0, stream>>>(state_costs_d, du_d, nu_d, normalizer, num_timesteps);
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//float getSolutionCost(COSTS_T *mppi_costs, float* trajectory, int num_timesteps)
//{
//  float cost;
//  float* trajectory_d;
//  float* costs_d;
//  float* costs = new float[num_timesteps];
//  HANDLE_ERROR( cudaMalloc((void**)&trajectory_d, num_timesteps*STATE_DIM*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&costs_d, num_timesteps*sizeof(float)));
//  HANDLE_ERROR(cudaMemcpy(trajectory_d, trajectory, num_timesteps*STATE_DIM*sizeof(float), cudaMemcpyHostToDevice));
//  solutionCostKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y><<<1, num_timesteps>>>(mppi_costs, trajectory_d, costs_d, num_timesteps);
//  CudaCheckError();
//  HANDLE_ERROR( cudaDeviceSynchronize() );
//  HANDLE_ERROR(cudaMemcpy(costs, costs_d, num_timesteps*sizeof(float), cudaMemcpyDeviceToHost));
//  for (int i = 0; i < num_timesteps; i++){
//    cost += costs[i];
//  }
//  return cost;
//}
//
//#undef BLOCKSIZE_X
//#undef BLOCKSIZE_Y
//#undef BLOCKSIZE_WRX
//#undef STATE_DIM
//#undef CONTROL_DIM
//#undef SHARED_MEM_REQUEST_GRD
//#undef SHARED_MEM_REQUEST_BLK
//#undef NUM_ROLLOUTS
//
//
///******************************************************************************************************************
//MPPI Controller implementation
//*******************************************************************************************************************/
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::RobustMPPIController(DYNAMICS_T* model, COSTS_T* costs,
//                                                                              int num_timesteps, int hz, float gamma,
//                                                                              float* exploration_var, float* init_u,
//                                                                              int num_optimization_iters, int opt_stride,
//                                                                              cudaStream_t stream)
//{
//  //Initialize internal classes which use the CUDA API.
//  model_ = model;
//  costs_ = costs;
//  curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
//  curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL);
//
//  //Set the CUDA stream, and attach unified memory to the particular stream.
//  //This must be done AFTER all internal classes that use unified memory are initialized (cost and model)
//  setCudaStream(stream);
//
//  //Initialize parameters, including the number of rollouts and timesteps
//  hz_ = hz;
//  numTimesteps_ = num_timesteps;
//  optimizationStride_ = opt_stride;
//  gamma_ = gamma;
//  num_iters_ = num_optimization_iters;
//
//  //Initialize host vectors
//  nu_.assign(exploration_var, exploration_var + CONTROL_DIM);
//  init_u_.assign(init_u, init_u + CONTROL_DIM);
//  importance_hist_.assign(2*CONTROL_DIM, 0);
//  optimal_control_hist_.assign(2*CONTROL_DIM, 0);
//  state_solution_.assign(numTimesteps_*STATE_DIM, 0);
//  control_solution_.assign(numTimesteps_*CONTROL_DIM, 0);
//  du_.resize(numTimesteps_*CONTROL_DIM);
//  feedback_gains_.assign(numTimesteps_*CONTROL_DIM*STATE_DIM, 0);
//
//  U_.resize(numTimesteps_*CONTROL_DIM);
//  U_optimal_.resize(numTimesteps_*CONTROL_DIM);
//
//  augmented_nominal_costs_.resize(NUM_ROLLOUTS);
//  augmented_real_costs_.resize(NUM_ROLLOUTS);
//  pure_real_costs_.resize(NUM_ROLLOUTS);
//
//  //Allocate memory on the device.
//  allocateCudaMem();
//  //Transfer exploration variance to device.
//  HANDLE_ERROR(cudaMemcpyAsync(nu_d_, nu_.data(), CONTROL_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));
//  //Get the parameters for the control input and initialize the sequence.
//  resetControls();
//  //Make sure all cuda operations have finished.
//  cudaStreamSynchronize(stream_);
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::~RobustMPPIController()
//{}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::setCudaStream(cudaStream_t stream)
//{
//  //Set the CUDA stream and attach unified memory in object to that stream
//  stream_ = stream;
//  model_->bindToStream(stream_);
//  costs_->bindToStream(stream_);
//  curandSetStream(gen_, stream_);
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::allocateCudaMem()
//{
//  HANDLE_ERROR( cudaMalloc((void**)&state_d_, 2*STATE_DIM*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&nominal_state_d_, STATE_DIM*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&augmented_nominal_costs_d_, NUM_ROLLOUTS*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&augmented_real_costs_d_, NUM_ROLLOUTS*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&pure_real_costs_d_, NUM_ROLLOUTS*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&U_d_, CONTROL_DIM*NUM_ROLLOUTS));
//  HANDLE_ERROR( cudaMalloc((void**)&nu_d_, STATE_DIM*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&du_d_, 2*NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM*sizeof(float)));
//  HANDLE_ERROR( cudaMalloc((void**)&feedback_gains_d_, numTimesteps_*CONTROL_DIM*STATE_DIM*sizeof(float)));
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::deallocateCudaMem(){
//  cudaFree(state_d_);
//  cudaFree(nominal_state_d_);
//  cudaFree(nu_d_);
//  cudaFree(augmented_nominal_costs_d_);
//  cudaFree(augmented_real_costs_d_);
//  cudaFree(pure_real_costs_d_);
//  cudaFree(U_d_);
//  cudaFree(du_d_);
//  //Free cuda memory used by the model and costs.
//  model_->freeCudaMem();
//  costs_->freeCudaMem();
//  cudaStreamDestroy(stream_);
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::initDDP(ros::NodeHandle nh)
//{
//  util::DefaultLogger logger;
//  bool verbose = false;
//  ddp_model_ = new ModelWrapperDDP<DYNAMICS_T>(model_);
//  ddp_solver_ = new DDP<ModelWrapperDDP<DYNAMICS_T>>(1.0/hz_, numTimesteps_, 1, &logger, verbose);
//
//  Q_.setIdentity();
//  Q_.diagonal() << nvr_control::getRosParam<double>("Qx", nh),
//                   nvr_control::getRosParam<double>("Qy", nh),
//                   nvr_control::getRosParam<double>("Qyaw", nh),
//                   nvr_control::getRosParam<double>("Qr", nh),
//                   nvr_control::getRosParam<double>("Qvx", nh),
//                   nvr_control::getRosParam<double>("Qvy", nh),
//                   nvr_control::getRosParam<double>("Qnyawd", nh);
//  std::cout << Q_ << std::endl;
//
//  Qf_.setIdentity();
//  Qf_.diagonal() << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
//
//  R_.setIdentity();
//  R_.diagonal() << nvr_control::getRosParam<double>("Rsteer", nh), nvr_control::getRosParam<double>("Rthrottle", nh),
//
//  U_MIN_ << model_->control_rngs_[0].x, model_->control_rngs_[1].x;
//  U_MAX_ << model_->control_rngs_[0].y, model_->control_rngs_[1].y;
//
//  //Define the running and terminal cost
//  run_cost_ = new TrackingCostDDP<ModelWrapperDDP<DYNAMICS_T>>(Q_, R_, numTimesteps_);
//  terminal_cost_ = new TrackingTerminalCost<ModelWrapperDDP<DYNAMICS_T>>(Qf_);
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::computeFeedbackGains(Eigen::MatrixXf state)
//{
//  Eigen::MatrixXf control_traj = Eigen::MatrixXf::Zero(CONTROL_DIM, numTimesteps_);
//  for (int t = 0; t < numTimesteps_; t++){
//    for (int i = 0; i < CONTROL_DIM; i++){
//      control_traj(i,t) = U_[CONTROL_DIM*t + i];
//    }
//  }
//  run_cost_->setTargets(state_solution_.data(), U_.data(), numTimesteps_);
//  terminal_cost_->xf = run_cost_->traj_target_x_.col(numTimesteps_ - 1);
//  result_ = ddp_solver_->run(state, control_traj, *ddp_model_, *run_cost_, *terminal_cost_, U_MIN_, U_MAX_);
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//OptimizerResult<ModelWrapperDDP<DYNAMICS_T>> RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::getFeedbackGains()
//{
//  return result_;
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::resetControls()
//{
//  int i,j;
//  //Set all the control values to their initial settings.
//  for (i = 0; i < numTimesteps_; i++) {
//    for (j = 0; j < CONTROL_DIM; j++) {
//      U_[i*CONTROL_DIM + j] = init_u_[j];
//    }
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::cutThrottle()
//{
//  costs_->params_.desired_speed = 0.0;
//  model_->control_rngs_[1].y = 0.0; //Max throttle to zero
//  costs_->paramsToDevice();
//  model_->paramsToDevice();
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::savitskyGolay()
//{
//  int i,j;
//  Eigen::MatrixXf filter(1,5);
//  Eigen::MatrixXf U_smoothed = Eigen::MatrixXf::Zero(8 + numTimesteps_, CONTROL_DIM);
//
//  filter << -3, 12, 17, 12, -3;
//  filter /= 35.0;
//
//  //Smooth the importance sampler
//  for (i = 0; i < numTimesteps_ + 4; i++){
//    if (i < 2) {
//      for (j = 0; j < CONTROL_DIM; j++){
//        U_smoothed(i, j) = importance_hist_[CONTROL_DIM*i + j];
//      }
//    }
//    else if (i < numTimesteps_ + 2) {
//      for (j = 0; j < CONTROL_DIM; j++){
//        U_smoothed(i,j) = U_[CONTROL_DIM*(i - 2) + j];
//      }
//    }
//    else{
//      for (j = 0; j < CONTROL_DIM; j++) {
//        U_smoothed(i, j) = U_[CONTROL_DIM*(numTimesteps_ - 1) + j];
//      }
//    }
//  }
//  for (i = 0; i < numTimesteps_; i++){
//    for (j = 0; j < CONTROL_DIM; j++){
//      U_[CONTROL_DIM*i + j] = (filter*U_smoothed.block<5,1>(i,j))(0,0);
//    }
//  }
//
//  //Smooth the optimal control
//  for (i = 0; i < numTimesteps_ + 4; i++){
//    if (i < 2) {
//      for (j = 0; j < CONTROL_DIM; j++){
//        U_smoothed(i, j) = optimal_control_hist_[CONTROL_DIM*i + j];
//      }
//    }
//    else if (i < numTimesteps_ + 2) {
//      for (j = 0; j < CONTROL_DIM; j++){
//        U_smoothed(i,j) = U_optimal_[CONTROL_DIM*(i - 2) + j];
//      }
//    }
//    else{
//      for (j = 0; j < CONTROL_DIM; j++) {
//        U_smoothed(i, j) = U_optimal_[CONTROL_DIM*(numTimesteps_ - 1) + j];
//      }
//    }
//  }
//  for (i = 0; i < numTimesteps_; i++){
//    for (j = 0; j < CONTROL_DIM; j++){
//      U_optimal_[CONTROL_DIM*i + j] = (filter*U_smoothed.block<5,1>(i,j))(0,0);
//    }
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::computeNominalTraj(Eigen::Matrix<float, STATE_DIM, 1> state)
//{
//  int i,j;
//  Eigen::MatrixXf s(7,1);
//  Eigen::MatrixXf u(2,1);
//  s = state;
//  for (i = 0; i < numTimesteps_; i++){
//    for (j = 0; j < STATE_DIM; j++){
//      //Set the current state solution
//      state_solution_[i*STATE_DIM + j] = s(j);
//    }
//    u << U_[2*i], U_[2*i + 1];
//    model_->updateState(s,u);
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::updateImportanceSampler(Eigen::Matrix<float, STATE_DIM, 1> state, int stride)
//{
//  int real_stride = stride;
//
//  //If this is the first iteration, just set the nominal state to the current state
//  if (!nominalStateInit_){
//    for (int i = 0; i < STATE_DIM; i++){
//      nominal_state_(i) = state(i);
//    }
//    nominalStateInit_ = true;
//    stride = 0;
//  }
//  else{ //Otherwise determine the optimal values for the nominal state and stride
//    if (SAMPLES_PER_CONDITION % BDIM_X != 0){
//      ROS_FATAL("SAMPLES_PER_CONDITION not multiple of BDIM_X!");
//    }
//    else if (NUM_CANDIDATES % 2 == 0){
//      ROS_FATAL("NUM_CANDIDATES must be odd!");
//    }
//
//    //Create the "triangle" of candidate initial conditions
//    std::vector<int> IsStrides(NUM_CANDIDATES);
//    std::vector<float> IsStates(STATE_DIM*NUM_CANDIDATES); ///< Array of the trajectory costs.
//    std::vector<float> IsCosts(NUM_CANDIDATES*SAMPLES_PER_CONDITION);
//    std::vector<float> IsTotalCosts(NUM_CANDIDATES);
//    std::vector<float> alpha_weights(3*NUM_CANDIDATES);
//    for (int i = 0; i < NUM_CANDIDATES/2 + 1; i++){
//      alpha_weights[i] = 1 - i/4.0;
//      alpha_weights[NUM_CANDIDATES + i] = i/4.0;
//      alpha_weights[2*NUM_CANDIDATES + i] = 0.0;
//    }
//    for (int i = 1; i < NUM_CANDIDATES/2 + 1; i++){
//      alpha_weights[NUM_CANDIDATES/2 + i] = 0.0;
//      alpha_weights[NUM_CANDIDATES + NUM_CANDIDATES/2 + i] = 1 - i/4.0;
//      alpha_weights[2*NUM_CANDIDATES + NUM_CANDIDATES/2 + i] = i/4.0;
//    }
//    for (int i = 0; i < NUM_CANDIDATES; i++){
//      for (int j = 0; j < STATE_DIM; j++){
//        IsStates[STATE_DIM*i + j] = alpha_weights[i]*state_solution_[j];
//        IsStates[STATE_DIM*i + j] += alpha_weights[NUM_CANDIDATES + i]*state_solution_[STATE_DIM*stride + j];
//        IsStates[STATE_DIM*i + j] += alpha_weights[2*NUM_CANDIDATES + i]*state(j);
//      }
//      IsStrides[i] = round(alpha_weights[i]*0 + alpha_weights[NUM_CANDIDATES + i]*stride + alpha_weights[2*NUM_CANDIDATES + i]*stride);
//    }
//
//    //Launch the kernel to compute the expected cost of each initial condition candidate
//    float *IsStates_d, *IsCosts_d;
//    int *IsStrides_d;
//    HANDLE_ERROR( cudaMalloc((void**)&IsStates_d, NUM_CANDIDATES*STATE_DIM*sizeof(float)));
//    HANDLE_ERROR( cudaMalloc((void**)&IsStrides_d, NUM_CANDIDATES*sizeof(int)));
//    HANDLE_ERROR( cudaMalloc((void**)&IsCosts_d, NUM_CANDIDATES*SAMPLES_PER_CONDITION*sizeof(float)));
//    HANDLE_ERROR( cudaMemcpyAsync(IsStates_d, IsStates.data(), NUM_CANDIDATES*STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));
//    HANDLE_ERROR( cudaMemcpyAsync(IsStrides_d, IsStrides.data(), NUM_CANDIDATES*sizeof(float), cudaMemcpyHostToDevice, stream_));
//    HANDLE_ERROR( cudaMemcpyAsync(U_d_, U_.data(), CONTROL_DIM*numTimesteps_*sizeof(float), cudaMemcpyHostToDevice, stream_));
//
//    curandGenerateNormal(gen_, du_d_, SAMPLES_PER_CONDITION*NUM_CANDIDATES*numTimesteps_*CONTROL_DIM, 0.0, 1.0);
//
//    launchInitEvalKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>(numTimesteps_, IsStates_d, IsStrides_d, U_d_, du_d_, nu_d_, IsCosts_d,
//                                                                        SAMPLES_PER_CONDITION, model_, costs_, optimizationStride_, stream_);
//    HANDLE_ERROR( cudaMemcpyAsync(IsCosts.data(), IsCosts_d, NUM_CANDIDATES*SAMPLES_PER_CONDITION*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    cudaStreamSynchronize(stream_);
//
//    //Compute the mass of each initial condition candidate
//    float baseline = IsCosts[0];
//    for (int i = 0; i < SAMPLES_PER_CONDITION; i++){
//      if (IsCosts[i] < baseline){
//        baseline = IsCosts[i];
//      }
//    }
//    for (int i = 0; i < NUM_CANDIDATES; i++){
//      IsTotalCosts[i] = 0;
//      for (int j = 0; j < SAMPLES_PER_CONDITION; j++){
//        IsTotalCosts[i] += expf(-gamma_*(IsCosts[i*SAMPLES_PER_CONDITION + j] - baseline));
//      }
//    }
//    for (int i = 0; i < NUM_CANDIDATES; i++){
//      IsTotalCosts[i] /= (1.0*SAMPLES_PER_CONDITION);
//    }
//
//    for (int i = 0; i < NUM_CANDIDATES; i++){
//      IsTotalCosts[i] = -1.0/gamma_ *logf(IsTotalCosts[i]) + baseline;
//    }
//
//    //Now get the closest initial condition that is above the threshold.
//    int bestIdx = 0;
//    for (int i = 1; i < NUM_CANDIDATES; i++){
//      if (IsTotalCosts[i] < value_func_threshold_){
//        bestIdx = i;
//      }
//    }
//    status_.mode = bestIdx;
//    stride = IsStrides[bestIdx];
//    for (int i = 0; i < STATE_DIM; i++){
//      nominal_state_(i) = IsStates[STATE_DIM*bestIdx + i];
//    }
//    cudaFree(IsStates_d);
//    cudaFree(IsStrides_d);
//    cudaFree(IsCosts_d);
//  }
//
//  //Save the importance sampler history
//  if (stride == 1){
//    importance_hist_[0] = importance_hist_[2];
//    importance_hist_[1] = importance_hist_[3];
//    importance_hist_[2] = U_[0];
//    importance_hist_[3] = U_[1];
//  }
//  else if (stride > 0){
//    int t = stride - 2;
//    for (int i = 0; i < 4; i++){
//      importance_hist_[i] = U_[t + i];
//    }
//  }
//  //Save the optimal control history
//  if (real_stride == 1){
//    optimal_control_hist_[0] = optimal_control_hist_[2];
//    optimal_control_hist_[1] = optimal_control_hist_[3];
//    optimal_control_hist_[2] = U_optimal_[0];
//    optimal_control_hist_[3] = U_optimal_[1];
//  }
//  else if (real_stride > 0){
//    int t = real_stride - 2;
//    for (int i = 0; i < 4; i++){
//      optimal_control_hist_[i] = U_optimal_[t + i];
//    }
//  }
//
//  //Update the importance sampling trajectory
//  for (int i = 0; i < numTimesteps_- stride; i++) {
//    for (int j = 0; j < CONTROL_DIM; j++) {
//      U_[i*CONTROL_DIM + j] = U_[(i+stride)*CONTROL_DIM + j];
//    }
//  }
//  for (int j = 1; j <= stride; j++) {
//    for (int i = 0; i < CONTROL_DIM; i++){
//      U_[(numTimesteps_ - j)*CONTROL_DIM + i] = init_u_[i];
//    }
//  }
//
//  //Lastly compute the nominal trajectory and feedback gains
//  computeNominalTraj(nominal_state_);
//  computeFeedbackGains(nominal_state_);
//  for (int t = 0; t < numTimesteps_ - stride; t++){
//    for (int i = 0; i < CONTROL_DIM; i++){
//      for (int j = 0; j < STATE_DIM; j++){
//        feedback_gains_[t*STATE_DIM*CONTROL_DIM + i*STATE_DIM + j] = result_.feedback_gain[t](i,j);
//      }
//    }
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//void RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::computeControl(Eigen::Matrix<float, STATE_DIM, 1> state)
//{
//  //First transfer the state and current control sequence to the device.
//  costs_->paramsToDevice();
//  model_->paramsToDevice();
//  float std_err, val, mean;
//  //Transfer the state and nominal state to state_d on the GPU
//  HANDLE_ERROR( cudaMemcpyAsync(feedback_gains_d_, feedback_gains_.data(), numTimesteps_*CONTROL_DIM*STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));
//  //Real state is the first portion of the augmented state
//  HANDLE_ERROR( cudaMemcpyAsync(state_d_, state.data(), STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));
//  //Nominal state is the second part of the augmented state
//  HANDLE_ERROR( cudaMemcpyAsync(state_d_ + STATE_DIM, nominal_state_.data(), STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));
//
//  for (int opt_iter = 0; opt_iter < num_iters_; opt_iter++) {
//    HANDLE_ERROR( cudaMemcpyAsync(U_d_, U_.data(), CONTROL_DIM*numTimesteps_*sizeof(float), cudaMemcpyHostToDevice, stream_));
//    //Generate a bunch of random numbers for control deviations
//    curandGenerateNormal(gen_, du_d_, NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM, 0.0, 1.0);
//    //Make a second copy of the random deviations
//    HANDLE_ERROR( cudaMemcpyAsync(du_d_ + NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM, du_d_, NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM*sizeof(float), cudaMemcpyDeviceToDevice, stream_));
//
//    //Launch the augmented importance sampling rollout kernel
//    launchRolloutKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>(numTimesteps_, state_d_, feedback_gains_d_, U_d_, du_d_,
//                                                                       nu_d_, augmented_nominal_costs_d_, augmented_real_costs_d_,
//                                                                       pure_real_costs_d_, model_, costs_, 1.0/gamma_,
//                                                                       optimizationStride_, stream_);
//
//    HANDLE_ERROR(cudaMemcpyAsync(augmented_nominal_costs_.data(), augmented_nominal_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    HANDLE_ERROR(cudaMemcpyAsync(augmented_real_costs_.data(), augmented_real_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    HANDLE_ERROR(cudaMemcpyAsync(pure_real_costs_.data(), pure_real_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    //Synchronize stream here since we want to do computations on the CPU
//    HANDLE_ERROR( cudaStreamSynchronize(stream_) );
//
//    //Compute the importance sampling update
//    for (int i = 0; i < NUM_ROLLOUTS; i++){
//      augmented_nominal_costs_[i] = 0.5*augmented_nominal_costs_[i] + 0.5*fmaxf(fminf(augmented_real_costs_[i],value_func_threshold_), augmented_nominal_costs_[i]);
//    }
//    float baseline = augmented_nominal_costs_[0];
//    for (int i = 0; i < NUM_ROLLOUTS; i++) {
//      if (augmented_nominal_costs_[i] < baseline){
//        baseline = augmented_nominal_costs_[i];
//      }
//    }
//    HANDLE_ERROR(cudaMemcpyAsync(augmented_nominal_costs_d_, augmented_nominal_costs_.data(), NUM_ROLLOUTS*sizeof(float), cudaMemcpyHostToDevice, stream_));
//    launchNormExpKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>(augmented_nominal_costs_d_, gamma_, baseline, stream_);
//    HANDLE_ERROR(cudaMemcpyAsync(augmented_nominal_costs_.data(), augmented_nominal_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    cudaStreamSynchronize(stream_);
//    normalizer_ = 0;
//    std_err = 0;
//    for (int i = 0; i < NUM_ROLLOUTS; i++) {
//      val = augmented_nominal_costs_[i];
//      normalizer_ += val;
//      std_err += val*val;
//    }
//    launchWeightedReductionKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>(augmented_nominal_costs_d_, du_d_ + NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM, nu_d_, normalizer_, numTimesteps_, stream_);
//    HANDLE_ERROR( cudaMemcpyAsync(U_.data(), du_d_ + NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM, numTimesteps_*CONTROL_DIM*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    cudaStreamSynchronize(stream_);
//    status_.V_nominal = -1.0/gamma_*logf(normalizer_/NUM_ROLLOUTS) + baseline;
//    mean = 1.0/NUM_ROLLOUTS*normalizer_;
//    std_err = 1.0/NUM_ROLLOUTS * std_err - mean*mean;
//    std_err = 1.0/gamma_*std_err/(mean*sqrtf(1.0*NUM_ROLLOUTS)) + 1.0/gamma_*std_err*std_err/(2*mean*mean*NUM_ROLLOUTS);
//    status_.mce_nominal = std_err;
//
//    //Compute the optimal control
//    baseline = pure_real_costs_[0];
//    for (int i = 0; i < NUM_ROLLOUTS; i++) {
//      if (pure_real_costs_[i] < baseline){
//        baseline = pure_real_costs_[i];
//      }
//    }
//    HANDLE_ERROR(cudaMemcpyAsync(pure_real_costs_d_, pure_real_costs_.data(), NUM_ROLLOUTS*sizeof(float), cudaMemcpyHostToDevice, stream_));
//    launchNormExpKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>(pure_real_costs_d_, gamma_, baseline, stream_);
//    HANDLE_ERROR(cudaMemcpyAsync(pure_real_costs_.data(), pure_real_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    cudaStreamSynchronize(stream_);
//    normalizer_ = 0;
//    std_err = 0;
//    for (int i = 0; i < NUM_ROLLOUTS; i++) {
//      val = pure_real_costs_[i];
//      normalizer_ += val;
//      std_err += val*val;
//    }
//    launchWeightedReductionKernel<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>(pure_real_costs_d_, du_d_, nu_d_, normalizer_, numTimesteps_, stream_);
//    HANDLE_ERROR( cudaMemcpyAsync(U_optimal_.data(), du_d_, numTimesteps_*CONTROL_DIM*sizeof(float), cudaMemcpyDeviceToHost, stream_));
//    cudaStreamSynchronize(stream_);
//    status_.V_actual = -1.0/gamma_*logf(normalizer_/NUM_ROLLOUTS) + baseline;
//    mean = 1.0/NUM_ROLLOUTS*normalizer_;
//    std_err = 1.0/NUM_ROLLOUTS * std_err - mean*mean;
//    std_err = 1.0/gamma_*std_err/(mean*sqrtf(1.0*NUM_ROLLOUTS)) + 1.0/gamma_*std_err*std_err/(2*mean*mean*NUM_ROLLOUTS);
//    status_.mce_actual = std_err;
//  }
//
//  //Smooth for the next optimization round
//  savitskyGolay();
//
//  //Update the optimal control solution
//  for (int i = 0; i < numTimesteps_; i++){
//    for (int j = 0; j < CONTROL_DIM; j++){
//      control_solution_[i*CONTROL_DIM + j] = fminf(1.0, fmaxf(-1.0, U_optimal_[i*CONTROL_DIM + j]));
//    }
//  }
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//std::vector<float> RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::getControlSeq()
//{
//  return control_solution_;
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//std::vector<float> RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::getStateSeq()
//{
//  return state_solution_;
//}
//
//template<class DYNAMICS_T, class COSTS_T, int ROLLOUTS, int BDIM_X, int BDIM_Y>
//TubeDiagnostics RobustMPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::getTubeDiagnostics()
//{
//  return status_;
//}
