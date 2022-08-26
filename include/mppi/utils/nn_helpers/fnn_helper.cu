//
// Created by jason on 8/18/22.
//

#include "fnn_helper.cuh"

template <class PARAMS_T>
FNNHelper<PARAMS_T>::FNNHelper<PARAMS_T>(cudaStream_t stream) : Managed(stream)
{
  CPUSetup();
}

template <class PARAMS_T>
FNNHelper<PARAMS_T>::FNNHelper<PARAMS_T>(std::string model_path, cudaStream_t stream) : FNNHelper<PARAMS_T>(stream)
{
  loadParams(model_path);
}

template <class PARAMS_T>
FNNHelper<PARAMS_T>::~FNNHelper()
{
  if (this->GPUMemStatus_)
  {
    freeCudaMem();
  }
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::loadParams(const std::string& model_path)
{
  int i, j, k;
  std::string bias_name = "";
  std::string weight_name = "";
  if (!fileExists(model_path))
  {
    std::cerr << "Could not load neural net model at path: " << model_path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(model_path);
  for (i = 0; i < NUM_LAYERS - 1; i++)
  {
    // NN index from 1
    bias_name = "dynamics_b" + std::to_string(i + 1);
    weight_name = "dynamics_W" + std::to_string(i + 1);

    cnpy::NpyArray weight_i_raw = param_dict[weight_name];
    cnpy::NpyArray bias_i_raw = param_dict[bias_name];
    double* weight_i = weight_i_raw.data<double>();
    double* bias_i = bias_i_raw.data<double>();

    // copy over the weights
    for (j = 0; j < this->params_.net_structure[i + 1]; j++)
    {
      for (k = 0; k < this->params_.net_structure[i]; k++)
      {
        // TODO why i - 1?
        this->params_.theta[this->params_.stride_idcs[2 * i] + j * this->params_.net_structure[i] + k] =
            (float)weight_i[j * this->params_.net_structure[i] + k];
        weights_[i](j, k) = (float)weight_i[j * this->params_.net_structure[i] + k];
      }
    }
    // copy over the bias
    for (j = 0; j < this->params_.net_structure[i + 1]; j++)
    {
      this->params_.theta[this->params_.stride_idcs[2 * i + 1] + j] = (float)bias_i[j];
      biases_[i](j, 0) = (float)bias_i[j];
    }
  }
  // Save parameters to GPU memory
  paramsToDevice();
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::loadParams(const cnpy::npz_t& param_dict)
{
  int i, j, k;
  std::string bias_name = "";
  std::string weight_name = "";
  for (i = 0; i < NUM_LAYERS - 1; i++)
  {
    // NN index from 1
    bias_name = "dynamics_b" + std::to_string(i + 1);
    weight_name = "dynamics_W" + std::to_string(i + 1);

    cnpy::NpyArray weight_i_raw = param_dict.at(weight_name);
    cnpy::NpyArray bias_i_raw = param_dict.at(bias_name);
    double* weight_i = weight_i_raw.data<double>();
    double* bias_i = bias_i_raw.data<double>();

    // copy over the weights
    for (j = 0; j < this->params_.net_structure[i + 1]; j++)
    {
      for (k = 0; k < this->params_.net_structure[i]; k++)
      {
        this->params_.theta[this->params_.stride_idcs[2 * i] + j * this->params_.net_structure[i] + k] =
            weight_i[j * this->params_.net_structure[i] + k];
        weights_[i](j, k) = weight_i[j * this->params_.net_structure[i] + k];
      }
    }
    // copy over the bias
    for (j = 0; j < this->params_.net_structure[i + 1]; j++)
    {
      this->params_.theta[this->params_.stride_idcs[2 * i + 1] + j] = bias_i[j];
      biases_[i](j, 0) = bias_i[j];
    }
  }
  // Save parameters to GPU memory
  paramsToDevice();
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::CPUSetup()
{
  // zeros out every matrix
  for (int i = 1; i < PARAMS_T::NUM_LAYERS; i++)
  {
    weighted_in_[i - 1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], 1);
    weights_[i - 1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], this->params_.net_structure[i - 1]);
    biases_[i - 1] = Eigen::MatrixXf::Zero(this->params_.net_structure[i], 1);
  }
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::updateModel(const std::vector<int>& description, const std::vector<float>& data)
{
  for (int i = 0; i < description.size(); i++)
  {
    if (description[i] != this->params_.net_structure[i])
    {
      std::cerr << "Invalid model trying to to be set for NN" << std::endl;
      exit(0);
    }
  }
  for (int i = 0; i < NUM_LAYERS - 1; i++)
  {
    for (int j = 0; j < this->params_.net_structure[i + 1]; j++)
    {
      for (int k = 0; k < this->params_.net_structure[i]; k++)
      {
        weights_[i](j, k) = data[this->params_.stride_idcs[2 * i] + j * this->params_.net_structure[i] + k];
        this->params_.theta[this->params_.stride_idcs[2 * i] + j * this->params_.net_structure[i] + k] =
            data[this->params_.stride_idcs[2 * i] + j * this->params_.net_structure[i] + k];
      }
    }
  }
  for (int i = 0; i < NUM_LAYERS - 1; i++)
  {
    for (int j = 0; j < this->params_.net_structure[i + 1]; j++)
    {
      biases_[i](j, 0) = data[this->params_.stride_idcs[2 * i + 1] + j];
      this->params_.theta[this->params_.stride_idcs[2 * i + 1] + j] = data[this->params_.stride_idcs[2 * i + 1] + j];
    }
  }
  if (this->GPUMemStatus_)
  {
    paramsToDevice();
  }
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    cudaFree(network_d_);
  }
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::GPUSetup()
{
  if (!this->GPUMemStatus_)
  {
    network_d_ = Managed::GPUSetup<FNNHelper<PARAMS_T>>(this);
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;
  }
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    // the only thing that should change is theta
    HANDLE_ERROR(cudaMemcpyAsync(this->network_d_->params_.theta, this->params_.theta, NUM_PARAMS * sizeof(float),
                                 cudaMemcpyHostToDevice));
  }
}

template <class PARAMS_T>
bool FNNHelper<PARAMS_T>::computeGrad(const Eigen::Ref<const input_array>& input, Eigen::Ref<dfdx> A)
{
  // compute forward to see gradient values
  output_array output;
  forward(input, output);
  return computeGrad(A);
}

template <class PARAMS_T>
bool FNNHelper<PARAMS_T>::computeGrad(Eigen::Ref<dfdx> A)
{
  // Start backprop
  Eigen::MatrixXf ip_delta = Eigen::MatrixXf::Identity(OUTPUT_DIM, OUTPUT_DIM);

  // Main backprop loop
  for (int i = NUM_LAYERS - 2; i > 0; i--)
  {
    Eigen::MatrixXf zp = weighted_in_[i - 1];
    for (int j = 0; j < this->params_.net_structure[i]; j++)
    {
      zp(j) = TANH_DERIV(zp(j));
    }
    ip_delta = ((weights_[i]).transpose() * ip_delta).eval();
    for (int j = 0; j < OUTPUT_DIM; j++)
    {
      ip_delta.col(j) = ip_delta.col(j).array() * zp.array();
    }
  }

  ip_delta = (((weights_[0]).transpose()) * ip_delta).transpose().eval();
  for (int i = 0; i < INPUT_DIM; i++)
  {
    A.col(i) = ip_delta.col(i);
  }
  return true;
}

template <class PARAMS_T>
void FNNHelper<PARAMS_T>::forward(const Eigen::Ref<const input_array>& input, Eigen::Ref<output_array> output)
{
  int i, j;
  Eigen::MatrixXf acts = input;
  for (int i = 0; i < NUM_LAYERS - 1; i++)
  {
    weighted_in_[i] = (weights_[i] * acts + biases_[i]).eval();
    acts = Eigen::MatrixXf::Zero(this->params_.net_structure[i + 1], 1);
    if (i < NUM_LAYERS - 2)
    {  // Last layer doesn't apply any non-linearity
      for (j = 0; j < this->params_.net_structure[i + 1]; j++)
      {
        acts(j) = TANH((weighted_in_[i])(j));  // Nonlinear component.
      }
    }
    else
    {
      for (j = 0; j < this->params_.net_structure[i + 1]; j++)
      {
        acts(j) = (weighted_in_[i])(j);
      }
    }
  }
  output = acts;
}

template <class PARAMS_T>
__device__ void FNNHelper<PARAMS_T>::initialize(float* theta_s)
{
  static_assert(std::is_trivially_copyable<PARAMS_T>::value);
  PARAMS_T* shared_params = (PARAMS_T*)theta_s;
  *shared_params = this->params_;
}

template <class PARAMS_T>
__device__ float* FNNHelper<PARAMS_T>::forward(float* input, float* theta_s, PARAMS_T* params, int shift)
{
  float* curr_act;
  float* next_act;
  float* tmp_act;
  float tmp;
  float* W;
  float* b;
  uint tdx = threadIdx.x;
  uint tdy = threadIdx.y;
  uint tdz = threadIdx.z;
  uint i, j, k;
  curr_act = &theta_s[shift];
  next_act = &theta_s[shift + LARGEST_LAYER];
  // iterate through the part of the state that should be an input to the NN
  if (input != nullptr)
  {
    for (i = tdy; i < INPUT_DIM; i += blockDim.y)
    {
      curr_act[i] = input[i];
    }
  }
  __syncthreads();
  // iterate through each layer
  for (i = 0; i < NUM_LAYERS - 1; i++)
  {
    W = &params->theta[params->stride_idcs[2 * i]];      // weights
    b = &params->theta[params->stride_idcs[2 * i + 1]];  // biases

    // for first non input layer until last layer this thread deals with
    // calculates the next activation based on current
    for (j = tdy; j < params->net_structure[i + 1]; j += blockDim.y)
    {
      tmp = 0;
      // apply each neuron activation from current layer
      for (k = 0; k < params->net_structure[i]; k++)
      {
        // No atomic add necessary.
        tmp += W[j * params->net_structure[i] + k] * curr_act[k];
      }
      // add bias from next layer and neuron
      tmp += b[j];
      if (i < NUM_LAYERS - 2)
      {
        tmp = TANH(tmp);
      }
      next_act[j] = tmp;
    }
    // Swap the two pointers
    tmp_act = curr_act;
    curr_act = next_act;
    next_act = tmp_act;
    __syncthreads();
  }
  return curr_act;
}

template <class PARAMS_T>
__device__ float* FNNHelper<PARAMS_T>::forward(float* input, float* theta_s)
{
  uint tdx = threadIdx.x;
  uint tdz = threadIdx.z;
  PARAMS_T* params = (PARAMS_T*)theta_s;
  return forward(input, theta_s, params, SHARED_MEM_REQUEST_GRD + (2 * LARGEST_LAYER) * (blockDim.x * tdz + tdx));
}
