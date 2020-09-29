/*
 * Created on Sun Sep 28 2020 by Bogdan
 */

#ifndef FEEDBACK_CONTROLLERS_DDP_CUH_
#define FEEDBACK_CONTROLLERS_DDP_CUH_

#include <mppi/feedback_controllers/feedback.cuh>

// Class where methods are implemented
template <class CLASS_T, class DYN_T>
class GPU_DDP : public GPUFeedbackController<CLASS_T, DYN_T> {
public:
  GPU_DDP();

  void k();
  void copyToDevice
}

// Alias class which
// template <class DYN_T>
// class BELUGA : public GPU_DDP<BELUGA<DYN_T>, DYN_T> {
// public:
//   BELUGA(cudaStream_t stream=0) : GPU_DDP<BELUGA<DYN_T>, DYN_T>(stream) {};
// }


template <class DYN_T, int NUM_TIMESTEPS>
class DDPFeedback : public FeedbackController<DYN_T, GPU_DDP<GPU_DDP<>, DYN_T>, NUM_TIMESTEPS> {

}

#endif  // FEEDBACK_CONTROLLERS_DDP_CUH_
