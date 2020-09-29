/*
 * Created on Sun Sep 28 2020 by Bogdan
 */

#ifndef FEEDBACK_CONTROLLERS_DDP_CUH_
#define FEEDBACK_CONTROLLERS_DDP_CUH_

#include <mppi/feedback_controllers/feedback.cuh>
#include <mppi/ddp/ddp.h>

// Class where methods are implemented
template <class CLASS_T, class DYN_T>
class DeviceDDPImpl : public GPUFeedbackController<CLASS_T, DYN_T> {
public:
  DeviceDDPImpl(int num_timesteps, cudaStream_t stream = 0);

  void allocateCUDAMemory();
  void deallocateCUDAMemory();

  void k(const float * x_act, const float * x_goal,
         const float t, float * theta,
         float* control_output);
  void copyToDevice();
  // Nothing to copy back
  void copyFromDevice() {}
protected:
  float * fb_gains_ = nullptr;
  // Needed for allocating memory for feedback gains
  int num_timesteps_ = 0;
}

// Alias class for
template <class DYN_T>
class DeviceDDP : public DeviceDDPImpl<DeviceDDP<DYN_T>, DYN_T> {
public:
  DeviceDDP(cudaStream_t stream=0) : DeviceDDPImpl<DeviceDDP<DYN_T>, DYN_T>(stream) {};
}


template <class DYN_T, int NUM_TIMESTEPS>
class DDPFeedback : public FeedbackController<DeviceDDP<DYN_T>, NUM_TIMESTEPS> {

}

#ifdef __CUDACC__
#include "ddp.cu"
#endif

#endif  // FEEDBACK_CONTROLLERS_DDP_CUH_
