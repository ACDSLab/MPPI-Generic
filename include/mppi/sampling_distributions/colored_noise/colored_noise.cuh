#pragma once
/**
 * Created by Bogdan, Dec 16, 2021
 * based off of https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py
 */

#include <cufft.h>

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

__global__ void floatToCuComplexArray(float* input, cuComplex* output, int length)
{
  for (int i = threadIdx.y; i < length; i++)
  {
    output[i] = make_cuComplex(input[i], input[length + i]);
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

void powerlaw_psd_gaussian(float exponent, int sample_length, int num_samples, int fmin = 0)
{
  std::vector<float> sample_freq;
  fftfreq(sample_length, sample_freq);
  float cutoff_freq = fmaxf(fmin, 1.0 / sample_length);
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
  sigma += powf(sample_freq.back() * (1.0 + (sample_length % 2) / 2.0), 2);
  sigma = sqrt(sigma) / sample_length;
  cufftComplex test = make_cuComplex(1.0, 2.0);
  // Sample the noise in frequency domain and reutrn to time domain
  cufftHandle plan;
  // Need (num_samples / 2 + 1) * num_samples of randomly sampled values
  for (int i = 0; i < num_samples; i++)
  {
    // Follow rough example from https://docs.nvidia.com/cuda/cufft/index.html#oned-real-to-complex-transforms
  }
  HANDLE_CUFFT_ERROR(cufftPlan1d(&plan, sample_freq.size(), CUFFT_C2R, num_samples));
  // HANDLE_CUFFT_ERROR(cufftSetStream(plan, stream));
  // freq_data needs to be num_samples number of sample_length/2 + 1 cuComplex values
  // time_data needs to be num_samples * sample_length floats
  // HANDLE_CUFFT_ERROR(cufftExecC2R(plan, freq_data, time_data));
  // HANDLE_ERROR(cudaStreamSyncrhonize(stream));
  // HANDLE_CUFFT_ERROR(cufftDestroy(plan));
}
