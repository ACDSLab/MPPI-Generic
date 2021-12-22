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

__global__ void configureFrequencyNoise(cufftComplex* noise, float* variance, int length, int num_samples)
{
  int sample_index = blockDim.x * blockIdx.x + threadIdx.x;
  int freq_index = blockDim.y * blockIdx.y + threadIdx.y;
  if (sample_index < num_samples && freq_index < length)
  {
    printf("Sample %d Fred %d: noise %f + i * %f, variance %f float version %f\n", sample_index, freq_index,
        noise[sample_index * length + freq_index].x, noise[sample_index * length + freq_index].y, variance[freq_index],
        ((float*) noise)[2 * (sample_index * length + freq_index)]);
    noise[sample_index * length + freq_index].x *= variance[freq_index];
    if (freq_index == 0)
    {
      noise[sample_index * length + freq_index].y = 0;
    }
    else if (length % 2 == 1 && freq_index == length - 1)
    {
      noise[sample_index * length + freq_index].y = 0;
    }
    else
    {
      noise[sample_index * length + freq_index].y *= variance[freq_index];
    }
    // for (int j = threadIdx.y; j < length; j += blockDim.y)
    // {
    //   noise[sample_index * length + j].x *= variance[j];
    //   noise[sample_index * length + j].y *= variance[j];
    // }
  }
}

__global__ void rearrangeNoise(float* input, float* output, float variance, int num_trajectories, int num_timesteps, int control_dim)
{
  int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  int time_index = blockIdx.y * blockDim.y + threadIdx.y;
  int control_index = blockIdx.z * blockDim.z + threadIdx.z;
  if (sample_index < num_trajectories && time_index < num_timesteps && control_index < control_dim)
  {
    output[(sample_index * num_timesteps + time_index) * control_dim + control_index] =
        input[(sample_index * control_dim + control_index) * num_timesteps + time_index] / variance;
    // printf("ROLLOUT %d TIME %d CONTROL %d: in %f out: %f\n", sample_index, time_index, control_index,
    //     input[(sample_index * control_dim + control_index) * num_timesteps + time_index],
    //     output[(sample_index * num_timesteps + time_index) * control_dim + control_index]);
  }
  // if (sample_index < num_trajectories) {
  //   for (int i = 0; i < num_timesteps; i++) {
  //     for (int j = 0; j < control_dim; j++) {
  //       output[sample_index * num_timesteps * control_dim + i * control_dim + j] = input[sample_index * num_timesteps
  //       * control_dim + j * num_timesteps + i];
  //     }
  //   }
  // }
}

void fftfreq(const int num_samples, std::vector<float>& result, const float spacing = 1)
{
  // result is of size floor(n/2) + 1
  int result_size = num_samples / 2 + 1;
  result.clear();
  result.resize(result_size);
  // std::cout << "for " << num_samples << " and " << spacing << " spacing" << std::endl;
  for (int i = 0; i < result_size; i++)
  {
    result[i] = i / (spacing * num_samples);
    // std::cout << i << ": " << result[i] << std::endl;
  }
}

void powerlaw_psd_gaussian(float exponent, int num_timesteps, int num_trajectories, int control_dim,
                           float* control_noise_d, curandGenerator_t& gen, cudaStream_t stream = 0, int fmin = 0)
{
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y = 32;
  const int BLOCKSIZE_Z = 1;
  std::vector<float> sample_freq;
  fftfreq(num_timesteps, sample_freq);
  float cutoff_freq = fmaxf(fmin, 1.0 / num_timesteps);
  std::cout << "Cutoff: " << cutoff_freq << " Variance before: ";
  for (int i = 0; i < sample_freq.size(); i++)
  {
    std::cout << sample_freq[i] << ", ";
    if (sample_freq[i] < cutoff_freq)
    {
      sample_freq[i] = cutoff_freq;
    }
    // std::cout << sample_freq[i] << ", ";
    sample_freq[i] = powf(sample_freq[i], -exponent / 2);
  }
  std::cout << std::endl;
  // Calculate variance
  float sigma = 0;
  std::for_each(sample_freq.begin(), sample_freq.end() - 1, [&sigma](float i) { sigma += powf(i, 2); });
  sigma += powf(sample_freq.back() * ((1.0 + (num_timesteps % 2)) / 2.0), 2);
  sigma = 2 * sqrt(sigma) / num_timesteps;
  std::cout << "UNIT VARIANCE: " << sigma << std::endl;
  cufftComplex test = make_cuComplex(1.0, 2.0);
  std::cout << "Variance: " << std::endl;
  for (int i = 0; i < sample_freq.size(); i++) {
    std::cout << sample_freq[i] << ", ";
  }
  std::cout << std::endl;
  // Sample the noise in frequency domain and reutrn to time domain
  cufftHandle plan;
  const int batch = num_trajectories * control_dim;
  // Need (num_timesteps / 2 + 1) * batch of randomly sampled values
  float* samples_in_freq_d;
  float* sigma_d;
  float* noise_in_time_d;
  cufftComplex* samples_in_freq_complex_d;
  HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_d, sizeof(float) * 2 * batch * sample_freq.size()));
  HANDLE_ERROR(cudaMalloc((void**)&sigma_d, sizeof(float) * sample_freq.size()));
  HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_complex_d, sizeof(cufftComplex) * batch * sample_freq.size()));
  HANDLE_ERROR(cudaMalloc((void**)&noise_in_time_d, sizeof(float) * batch * num_timesteps));
  // curandSetStream(gen, stream);
  auto status = curandGenerateNormal(gen, (float*)samples_in_freq_complex_d, 2 * batch * sample_freq.size(), 0.0, 1.0);
  // status = curandGenerateNormal(gen, samples_in_freq_d, 2 * batch * sample_freq.size(), 0.0, 1.0);
  if (status != CURAND_STATUS_SUCCESS) {
    std::cout << "ERROR: " << status << std::endl;
    exit(0);
  }
  // std::vector<float> noise_in_freq(2 * batch * sample_freq.size(), 0);
  // std::vector<cufftComplex> noise_in_complex(batch * sample_freq.size());
  // HANDLE_ERROR(cudaMemcpyAsync(noise_in_freq.data(), samples_in_freq_complex_d, sizeof(float) * noise_in_freq.size(), cudaMemcpyDeviceToHost, stream));
  // HANDLE_ERROR(cudaMemcpyAsync(noise_in_complex.data(), samples_in_freq_complex_d, sizeof(cufftComplex) * noise_in_complex.size(), cudaMemcpyDeviceToHost, stream));
  // cudaStreamSynchronize(stream);
  // std::cout << "Normally-distributed noise: ";
  // for (auto noise_i : noise_in_freq) {
  //   std::cout << noise_i << ", ";
  // }
  // std::cout << std::endl;

  // std::cout << "Normally-distributed complex: ";
  // for (auto complex_i : noise_in_complex) {
  //   std::cout << complex_i.x << ", " << complex_i.y << ", ";
  // }
  // std::cout << std::endl;

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaMemcpyAsync(sigma_d, sample_freq.data(), sizeof(float) * sample_freq.size(), cudaMemcpyHostToDevice, stream));
  // for (int i = 0; i < batch; i++)
  // {
  //   // Follow rough example from https://docs.nvidia.com/cuda/cufft/index.html#oned-real-to-complex-transforms
  // }
  const int variance_grid_x = (batch - 1) / BLOCKSIZE_X + 1;
  const int variance_grid_y = (sample_freq.size() - 1) / BLOCKSIZE_Y + 1;
  dim3 grid(variance_grid_x, variance_grid_y, 1);
  dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  // configureFrequencyNoise<<<grid, block, 0, stream>>>((cuComplex*) samples_in_freq_d, sigma_d, sample_freq.size(), batch);
  // std::cout << "Address outside of kernel: " << samples_in_freq_complex_d << std::endl;
  configureFrequencyNoise<<<grid, block, 0, stream>>>(samples_in_freq_complex_d, sigma_d, sample_freq.size(), batch);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_CUFFT_ERROR(cufftPlan1d(&plan, num_timesteps, CUFFT_C2R, batch));
  // HANDLE_CUFFT_ERROR(cufftPlan1d(&plan, sample_freq.size(), CUFFT_C2R, batch));
  HANDLE_CUFFT_ERROR(cufftSetStream(plan, stream));
  // freq_data needs to be batch number of num_timesteps/2 + 1 cuComplex values
  // time_data needs to be batch * num_timesteps floats
  HANDLE_CUFFT_ERROR(cufftExecC2R(plan, samples_in_freq_complex_d, noise_in_time_d));
  HANDLE_ERROR(cudaGetLastError());
  const int reorder_grid_x = (num_trajectories - 1) / BLOCKSIZE_X + 1;
  const int reorder_grid_y = (num_timesteps - 1) / BLOCKSIZE_Y + 1;
  const int reorder_grid_z = (control_dim - 1) / BLOCKSIZE_Z + 1;
  dim3 reorder_grid(reorder_grid_x, reorder_grid_y, reorder_grid_z);
  dim3 reorder_block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  HANDLE_ERROR(cudaGetLastError());
  // std::cout << "Grid: " << reorder_grid.x << ", " << reorder_grid.y << ", " << reorder_grid.z << std::endl;
  // std::cout << "Block: " << reorder_block.x << ", " << reorder_block.y << ", " << reorder_block.z << std::endl;
  rearrangeNoise<<<reorder_grid, reorder_block, 0, stream>>>(noise_in_time_d, control_noise_d, sigma, num_trajectories,
                                                             num_timesteps, control_dim);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_CUFFT_ERROR(cufftDestroy(plan));
  HANDLE_ERROR(cudaFree(samples_in_freq_d));
  HANDLE_ERROR(cudaFree(sigma_d));
  HANDLE_ERROR(cudaFree(samples_in_freq_complex_d));
  HANDLE_ERROR(cudaFree(noise_in_time_d));
}
