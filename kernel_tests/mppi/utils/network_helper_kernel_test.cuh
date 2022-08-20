//
// Created by jason on 8/19/22.
//

#ifndef MPPIGENERIC_NETWORK_HELPER_KERNEL_TEST_CUH
#define MPPIGENERIC_NETWORK_HELPER_KERNEL_TEST_CUH


template <class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
__global__ void parameterCheckTestKernel(NETWORK_T* model, float* theta, int* stride, int* net_structure)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0)
  {
    for (int i = 0; i < THETA_SIZE; i++)
    {
      theta[i] = model->getThetaPtr()[i];
    }
    for (int i = 0; i < STRIDE_SIZE; i++)
    {
      stride[i] = model->getStrideIdcsPtr()[i];
    }
    for (int i = 0; i < NUM_LAYERS; i++)
    {
      net_structure[i] = model->getNetStructurePtr()[i];
    }
  }
}

template <class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
void launchParameterCheckTestKernel(NETWORK_T& model, std::array<float, THETA_SIZE>& theta,
                                    std::array<int, STRIDE_SIZE>& stride, std::array<int, NUM_LAYERS>& net_structure)
{
  float* theta_d;
  int* stride_d;
  int* net_structure_d;

  HANDLE_ERROR(cudaMalloc((void**)&theta_d, sizeof(float) * theta.size()))
  HANDLE_ERROR(cudaMalloc((void**)&stride_d, sizeof(int) * stride.size()))
  HANDLE_ERROR(cudaMalloc((void**)&net_structure_d, sizeof(int) * net_structure.size()))

  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks(1, 1);
  parameterCheckTestKernel<NETWORK_T, THETA_SIZE, STRIDE_SIZE, NUM_LAYERS>
  <<<numBlocks, threadsPerBlock>>>(model.network_d_, theta_d, stride_d, net_structure_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(theta.data(), theta_d, sizeof(float) * theta.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(stride.data(), stride_d, sizeof(int) * stride.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(
          cudaMemcpy(net_structure.data(), net_structure_d, sizeof(int) * net_structure.size(), cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(theta_d);
  cudaFree(stride_d);
  cudaFree(net_structure_d);
}

template <typename NETWORK_T>
__global__ void forwardTestKernel(NETWORK_T* network, float* input, float* output, int num)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float theta[NETWORK_T::SHARED_MEM_REQUEST_GRD + NETWORK_T::SHARED_MEM_REQUEST_BLK];
  float* local_input = input + (tid * NETWORK_T::INPUT_DIM);
  float* local_output = output + (tid * NETWORK_T::OUTPUT_DIM);

  if (tid < num)
  {
    float* curr_act = network->forward(local_input, theta);
    for (int i = threadIdx.y; i < NETWORK_T::OUTPUT_DIM; i += blockDim.y)
    {
      output[i] = curr_act[i];
    }
    __syncthreads();
  }
}

template <typename NETWORK_T>
void launchForwardTestKernel(NETWORK_T& dynamics, std::vector<std::array<float, NETWORK_T::INPUT_DIM>>& input,
                          std::vector<std::array<float, NETWORK_T::OUTPUT_DIM>>& output, int dim_y)
{
  if (input.size() != output.size())
  {
    std::cerr << "Num States doesn't match num controls" << std::endl;
    return;
  }
  int count = input.size();
  float* input_d;
  float* output_d;
  HANDLE_ERROR(cudaMalloc((void**)&input_d, sizeof(float) * NETWORK_T::INPUT_DIM * count));
  HANDLE_ERROR(cudaMalloc((void**)&output_d, sizeof(float) * NETWORK_T::OUTPUT_DIM * count));

  HANDLE_ERROR(
          cudaMemcpy(input_d, input.data(), sizeof(float) * NETWORK_T::INPUT_DIM * count, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
          cudaMemcpy(output_d, input.data(), sizeof(float) * NETWORK_T::OUTPUT_DIM * count, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(count, dim_y);
  dim3 numBlocks(1, 1);
  forwardTestKernel<NETWORK_T><<<numBlocks, threadsPerBlock>>>(dynamics.network_d_, input_d, output_d, count);
  CudaCheckError();

  // Copy the memory back to the host
  HANDLE_ERROR(
          cudaMemcpy(input.data(), input_d, sizeof(float) * NETWORK_T::INPUT_DIM * count, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
          cudaMemcpy(output.data(), output_d, sizeof(float) * NETWORK_T::OUTPUT_DIM * count, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(input_d);
  cudaFree(output_d);
}

#endif  // MPPIGENERIC_NETWORK_HELPER_KERNEL_TEST_CUH
