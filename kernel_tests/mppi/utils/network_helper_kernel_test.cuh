//
// Created by jason on 8/19/22.
//

#ifndef MPPIGENERIC_NETWORK_HELPER_KERNEL_TEST_CUH
#define MPPIGENERIC_NETWORK_HELPER_KERNEL_TEST_CUH

// TODO check on multiple different blocks

template <class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
__global__ void parameterCheckTestKernel(NETWORK_T* model, float* theta, int* stride, int* net_structure,
                                         float* shared_theta, int* shared_stride, int* shared_net_structure)
{
  __shared__ float theta_s[NETWORK_T::SHARED_MEM_REQUEST_GRD/sizeof(float) + 1 + NETWORK_T::SHARED_MEM_REQUEST_BLK];
  model->initialize(theta_s);
  typename NETWORK_T::NN_PARAMS_T* params_shared = (typename NETWORK_T::NN_PARAMS_T*) theta_s;
  for (int i = 0; i < THETA_SIZE; i++)
  {
    theta[i] = model->getThetaPtr()[i];
    shared_theta[i] = params_shared->theta[i];
  }
  for (int i = 0; i < STRIDE_SIZE; i++)
  {
    stride[i] = model->getStrideIdcsPtr()[i];
    shared_stride[i] = params_shared->stride_idcs[i];
  }
  for (int i = 0; i < NUM_LAYERS; i++)
  {
    net_structure[i] = model->getNetStructurePtr()[i];
    shared_net_structure[i] = params_shared->net_structure[i];
  }
}

template <class NETWORK_T, int THETA_SIZE, int STRIDE_SIZE, int NUM_LAYERS>
void launchParameterCheckTestKernel(NETWORK_T& model, std::array<float, THETA_SIZE>& theta,
                                    std::array<int, STRIDE_SIZE>& stride, std::array<int, NUM_LAYERS>& net_structure,
                                    std::array<float, THETA_SIZE>& shared_theta,
                                    std::array<int, STRIDE_SIZE>& shared_stride, std::array<int, NUM_LAYERS>& shared_net_structure)
{
  float* theta_d;
  int* stride_d;
  int* net_structure_d;

  float* shared_theta_d;
  int* shared_stride_d;
  int* shared_net_structure_d;

  HANDLE_ERROR(cudaMalloc((void**)&theta_d, sizeof(float) * theta.size()))
  HANDLE_ERROR(cudaMalloc((void**)&stride_d, sizeof(int) * stride.size()))
  HANDLE_ERROR(cudaMalloc((void**)&net_structure_d, sizeof(int) * net_structure.size()))
  HANDLE_ERROR(cudaMalloc((void**)&shared_theta_d, sizeof(float) * theta.size()))
  HANDLE_ERROR(cudaMalloc((void**)&shared_stride_d, sizeof(int) * stride.size()))
  HANDLE_ERROR(cudaMalloc((void**)&shared_net_structure_d, sizeof(int) * net_structure.size()))

  dim3 threadsPerBlock(3, 1);
  dim3 numBlocks(10, 1);
  parameterCheckTestKernel<NETWORK_T, THETA_SIZE, STRIDE_SIZE, NUM_LAYERS>
  <<<numBlocks, threadsPerBlock>>>(model.network_d_, theta_d, stride_d, net_structure_d,
                                       shared_theta_d, shared_stride_d, shared_net_structure_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(theta.data(), theta_d, sizeof(float) * theta.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(stride.data(), stride_d, sizeof(int) * stride.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(
          cudaMemcpy(net_structure.data(), net_structure_d, sizeof(int) * net_structure.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(shared_theta.data(), theta_d, sizeof(float) * theta.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(shared_stride.data(), stride_d, sizeof(int) * stride.size(), cudaMemcpyDeviceToHost))
  HANDLE_ERROR(
          cudaMemcpy(shared_net_structure.data(), net_structure_d, sizeof(int) * net_structure.size(), cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(theta_d);
  cudaFree(stride_d);
  cudaFree(net_structure_d);
  cudaFree(shared_theta_d);
  cudaFree(shared_stride_d);
  cudaFree(shared_net_structure_d);
}

template <class NETWORK_T>
__global__ void parameterCheckTestKernel(NETWORK_T* model, typename NETWORK_T::LSTM_PARAMS_T* lstm_params,
                                         typename NETWORK_T::LSTM_PARAMS_T* shared_lstm_params,
                                         typename NETWORK_T::OUTPUT_PARAMS_T* fnn_params,
                                         typename NETWORK_T::OUTPUT_PARAMS_T* shared_fnn_params)
{
  __shared__ float theta_s[NETWORK_T::SHARED_MEM_REQUEST_GRD/sizeof(float) + 1 + NETWORK_T::SHARED_MEM_REQUEST_BLK];
  uint tid = blockIdx.x;

  *(lstm_params + tid) = model->getLSTMParams();
  *(fnn_params + tid) = model->getOutputModel()->getParams();

  model->initialize(theta_s);

  const int slide = NETWORK_T::LSTM_SHARED_MEM_GRD/sizeof(float) + 1;
  auto* fnn_params_shared =
          (typename NETWORK_T::OUTPUT_PARAMS_T*) (theta_s + slide);
  *(shared_fnn_params + tid) = *fnn_params_shared;

  auto* lstm_params_shared = (typename NETWORK_T::LSTM_PARAMS_T*) theta_s;
  *(shared_lstm_params + tid) = *lstm_params_shared;
}

template <class NETWORK_T>
void launchParameterCheckTestKernel(NETWORK_T& model, std::vector<typename NETWORK_T::LSTM_PARAMS_T>& lstm_params,
                                    std::vector<typename NETWORK_T::LSTM_PARAMS_T>& shared_lstm_params,
                                    std::vector<typename NETWORK_T::OUTPUT_PARAMS_T>& fnn_params,
                                    std::vector<typename NETWORK_T::OUTPUT_PARAMS_T>& shared_fnn_params)
{
  static_assert(NETWORK_T::SHARED_MEM_REQUEST_GRD != 0);

  typename NETWORK_T::LSTM_PARAMS_T* lstm_params_d = nullptr;
  typename NETWORK_T::LSTM_PARAMS_T* shared_lstm_params_d = nullptr;
  typename NETWORK_T::OUTPUT_PARAMS_T* fnn_params_d = nullptr;
  typename NETWORK_T::OUTPUT_PARAMS_T* shared_fnn_params_d = nullptr;

  int num = lstm_params.size();

  HANDLE_ERROR(cudaMalloc((void**)&lstm_params_d, sizeof(typename NETWORK_T::LSTM_PARAMS_T) * num));
  HANDLE_ERROR(cudaMalloc((void**)&shared_lstm_params_d, sizeof(typename NETWORK_T::LSTM_PARAMS_T) * num));
  HANDLE_ERROR(cudaMalloc((void**)&fnn_params_d, sizeof(typename NETWORK_T::OUTPUT_PARAMS_T) * num));
  HANDLE_ERROR(cudaMalloc((void**)&shared_fnn_params_d, sizeof(typename NETWORK_T::OUTPUT_PARAMS_T) * num));

  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks(num, 1);
  parameterCheckTestKernel<NETWORK_T>
  <<<numBlocks, threadsPerBlock>>>(model.network_d_, lstm_params_d, shared_lstm_params_d, fnn_params_d, shared_fnn_params_d);
  CudaCheckError();

  HANDLE_ERROR(cudaMemcpy(lstm_params.data(), lstm_params_d, sizeof(typename NETWORK_T::LSTM_PARAMS_T) * num, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(shared_lstm_params.data(), shared_lstm_params_d, sizeof(typename NETWORK_T::LSTM_PARAMS_T) * num, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(fnn_params.data(), fnn_params_d, sizeof(typename NETWORK_T::OUTPUT_PARAMS_T) * num, cudaMemcpyDeviceToHost))
  HANDLE_ERROR(cudaMemcpy(shared_fnn_params.data(), shared_fnn_params_d, sizeof(typename NETWORK_T::OUTPUT_PARAMS_T) * num, cudaMemcpyDeviceToHost))
  cudaDeviceSynchronize();

  cudaFree(lstm_params_d);
  cudaFree(shared_lstm_params_d);
  cudaFree(fnn_params_d);
  cudaFree(shared_fnn_params_d);
}

template <typename NETWORK_T, int BLOCKSIZE_X>
__global__ void forwardTestKernel(NETWORK_T* network, float* input, float* output, int num, int steps)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float theta_s[NETWORK_T::SHARED_MEM_REQUEST_GRD/sizeof(float)+1 + NETWORK_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X];
  float* local_input = input + (tid * NETWORK_T::INPUT_DIM);
  float* local_output = output + (tid * NETWORK_T::OUTPUT_DIM);

  network->initialize(theta_s);

  if (tid < num)
  {
    float* curr_act = nullptr;
    for(uint step = 0; step < steps; step++) {
      curr_act = network->forward(local_input, theta_s);
    }
    for (uint i = threadIdx.y; i < NETWORK_T::OUTPUT_DIM; i += blockDim.y)
    {
      local_output[i] = curr_act[i];
    }
    __syncthreads();
  }
}

template <typename NETWORK_T, int BLOCKSIZE_X>
void launchForwardTestKernel(NETWORK_T& dynamics, std::vector<std::array<float, NETWORK_T::INPUT_DIM>>& input,
                          std::vector<std::array<float, NETWORK_T::OUTPUT_DIM>>& output, int dim_y,
                             int steps = 1)
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

  const int gridsize_x = (count - 1) / BLOCKSIZE_X + 1;
  dim3 threadsPerBlock(BLOCKSIZE_X, dim_y);
  dim3 numBlocks(gridsize_x, 1);
  forwardTestKernel<NETWORK_T, BLOCKSIZE_X><<<numBlocks, threadsPerBlock>>>(dynamics.network_d_, input_d, output_d, count, steps);
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

template <typename NETWORK_T, int BLOCKSIZE_X>
__global__ void forwardTestKernelPreload(NETWORK_T* network, float* input, float* output, int num, int steps)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float theta_s[NETWORK_T::SHARED_MEM_REQUEST_GRD/sizeof(float)+1 + NETWORK_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X];
  float* local_input = input + (tid * NETWORK_T::INPUT_DIM);
  float* local_output = output + (tid * NETWORK_T::OUTPUT_DIM);

  network->initialize(theta_s);

  if (tid < num)
  {
    float* input_loc = network->getInputLocation(theta_s);
    for (int i = threadIdx.y; i < NETWORK_T::INPUT_DIM; i += blockDim.y)
    {
      input_loc[i] = local_input[i];
    }

    float* curr_act = nullptr;
    for(uint step = 0; step < steps; step++) {
      curr_act = network->forward(nullptr, theta_s);
    }
    for (uint i = threadIdx.y; i < NETWORK_T::OUTPUT_DIM; i += blockDim.y)
    {
      local_output[i] = curr_act[i];
    }
    __syncthreads();
  }
}

template <typename NETWORK_T, int BLOCKSIZE_X>
void launchForwardTestKernelPreload(NETWORK_T& dynamics, std::vector<std::array<float, NETWORK_T::INPUT_DIM>>& input,
                             std::vector<std::array<float, NETWORK_T::OUTPUT_DIM>>& output, int dim_y,
                             int steps = 1)
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

  const int gridsize_x = (count - 1) / BLOCKSIZE_X + 1;
  dim3 threadsPerBlock(BLOCKSIZE_X, dim_y);
  dim3 numBlocks(gridsize_x, 1);
  forwardTestKernelPreload<NETWORK_T, BLOCKSIZE_X><<<numBlocks, threadsPerBlock>>>(dynamics.network_d_, input_d, output_d, count, steps);
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
