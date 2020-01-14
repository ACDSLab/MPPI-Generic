#include "ar_nn_dynamics_kernel_test.cuh"


template <class NETWORK_T>
__global__ void parameterCheckTestKernel(NETWORK_T* model,  float* theta, int* stride, int* net_structure) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid == 0) {
    for(int i = 0; i < 1412; i++) {
      theta[i] = model->getThetaPtr()[i];
    }
    for(int i = 0; i < 6; i++) {
      stride[i] = model->getStrideIdcsPtr()[i];
    }
    for(int i = 0; i < 4; i++) {
      net_structure[i] = model->getNetStructurePtr()[i];
    }
  }
}

template<class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
void launchParameterCheckTestKernel(NETWORK_T& model, std::array<float, THETA_SIZE>& theta, std::array<int, STRIDE_SIZE>& stride,
                                    std::array<int, NUM_LAYERS>& net_structure) {
  float* theta_d;
  int* stride_d;
  int* net_structure_d;

  HANDLE_ERROR(cudaMalloc((void**)&theta_d, sizeof(float)*theta.size()))
  HANDLE_ERROR(cudaMalloc((void**)&stride_d, sizeof(int)*stride.size()))
  HANDLE_ERROR(cudaMalloc((void**)&net_structure_d, sizeof(int)*net_structure.size()))

  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks(1, 1);
  parameterCheckTestKernel<NETWORK_T><<<numBlocks,threadsPerBlock>>>(model.model_d_, theta_d, stride_d, net_structure_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(theta.data(), theta_d, sizeof(float)*theta.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(stride.data(), stride_d, sizeof(int)*stride.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(net_structure.data(), net_structure_d, sizeof(int)*net_structure.size(), cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(theta_d);
  cudaFree(stride_d);
  cudaFree(net_structure_d);
}


// explicit instantiation
template void launchParameterCheckTestKernel<NeuralNetModel<7,2,3,6,32,32,4>, 1412, 6, 4>(NeuralNetModel<7,2,3,6,32,32,4>& model, std::array<float, 1412>& theta, std::array<int, 6>& stride,
                                                                                          std::array<int, 4>& net_structure);

// explicit instantiation
template __global__ void parameterCheckTestKernel<NeuralNetModel<7,2,3,6,32,32,4>>(NeuralNetModel<7,2,3,6,32,32,4>* model,  float* theta, int* stride, int* net_structure);
