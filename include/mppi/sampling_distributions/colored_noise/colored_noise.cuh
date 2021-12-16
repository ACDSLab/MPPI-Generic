#pragma once
/**
 * Created by Bogdan, Dec 16, 2021
 * based off of https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py
 */

#include <cufft.h>
#include <curand.h>

#include <mppi/utils/gpu_err_chk.cuh>

#include <algorithm>
#include <iostream>
#include <vector>

const char* cufftGetErrorString(cufftResult& code)
{
  if (code == CUFFT_SUCCESS)
  {
    return "Success";
  }
  return "Error";
}

inline void cufftAssert(cufftResult code, const char* file, int line, bool abort = true)
{
  if (code != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFTassert: %s %s %d\n", cufftGetErrorString(code), file, line);
    if (abort)
    {
      exit(code);
    }
  }
}

#define HANDLE_CUFFT_ERROR(ans)                                                                                        \
  {                                                                                                                    \
    cufftAssert((ans), __FILE__, __LINE__);                                                                            \
  }

// __global__ void floatToCuComplexArray(float* input, float* variance, cuComplex* output, int length, int num_samples)
// {
//   for (int i = threadIdx.y; i < length; i++)
//   {
//     output[i] = make_cuComplex(input[i], input[length + i]);
//   }
// }

__global__ void configureFrequencyNoise(cuComplex* noise, float* variance, int length, int num_samples)
{
  int sample_index = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadIdx.x; i < num_samples; i += blockDim.x)
  {
    for (int j = threadIdx.y; j < length; j += blockDim.y)
    {
      noise[i * length + j].x *= variance[j];
      noise[i * length + j].y *= variance[j];
    }
  }
}

void fftfreq(const int num_samples, std::vector<float>& result, const float spacing = 1)
{
  // result is of size floor(n/2) + 1
  int result_size = num_samples / 2 + 1;
  result.clear();
  result.resize(result_size);
  std::cout << "for " << num_samples << " and " << spacing << " spacing" << std::endl;
  for (int i = 0; i < result_size; i++)
  {
    result[i] = i / (spacing * num_samples);
    std::cout << i << ": " << result[i] << std::endl;
  }
}

void powerlaw_psd_gaussian(float exponent, int num_timesteps, int num_trajectories, int control_dim,
                           curandGenerator_t gen, cudaStream_t stream = 0, int fmin = 0)
{
  std::vector<float> sample_freq;
  fftfreq(num_timesteps, sample_freq);
  float cutoff_freq = fmaxf(fmin, 1.0 / num_timesteps);
  for (int i = 0; i < sample_freq.size(); i++)
  {
    if (sample_freq[i] < cutoff_freq)
    {
      sample_freq[i] = cutoff_freq;
    }
    sample_freq[i] = powf(sample_freq[i], -exponent / 2);
  }
  // Calculate variance
  float sigma = 0;
  std::for_each(sample_freq.begin(), sample_freq.end() - 1, [&sigma](float i) { sigma += powf(i, 2); });
  sigma += powf(sample_freq.back() * (1.0 + (num_timesteps % 2) / 2.0), 2);
  sigma = sqrt(sigma) / num_timesteps;
  cufftComplex test = make_cuComplex(1.0, 2.0);
  // Sample the noise in frequency domain and reutrn to time domain
  cufftHandle plan;
  int batch = num_trajectories * control_dim;
  // Need (num_timesteps / 2 + 1) * batch of randomly sampled values
  float* samples_in_freq_d;
  float* sigma_d;
  cufftComplex* samples_in_freq_complex_d;
  HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_d, sizeof(float) * 2 * batch * sample_freq.size()));
  HANDLE_ERROR(cudaMalloc((void**)&sigma_d, sizeof(float) * sample_freq.size()));
  HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_complex_d, sizeof(cufftComplex) * batch * sample_freq.size()));
  // curandSetStream(gen, stream);
  curandGenerateNormal(gen, (float*)samples_in_freq_complex_d, batch * sample_freq.size(), 0.0, 1.0);
  for (int i = 0; i < batch; i++)
  {
    // Follow rough example from https://docs.nvidia.com/cuda/cufft/index.html#oned-real-to-complex-transforms
  }

  // configureFrequencyNoise<<<grid, block, 0, stream>>>(samples_in_freq_complex_d, sigma_d, sample_freq.size(), batch);
  HANDLE_CUFFT_ERROR(cufftPlan1d(&plan, sample_freq.size(), CUFFT_C2R, batch));
  HANDLE_CUFFT_ERROR(cufftSetStream(plan, stream));
  // freq_data needs to be batch number of num_timesteps/2 + 1 cuComplex values
  // time_data needs to be batch * num_timesteps floats
  // HANDLE_CUFFT_ERROR(cufftExecC2R(plan, samples_in_freq_complex_d, time_data));
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_CUFFT_ERROR(cufftDestroy(plan));
  HANDLE_ERROR(cudaFree(samples_in_freq_d));
  HANDLE_ERROR(cudaFree(sigma_d));
  HANDLE_ERROR(cudaFree(samples_in_freq_complex_d));
}
