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

__global__ void configureFrequencyNoise(cufftComplex* noise, float* variance, int length, int num_samples)
{
  int sample_index = blockDim.x * blockIdx.x + threadIdx.x;
  int freq_index = blockDim.y * blockIdx.y + threadIdx.y;
  if (sample_index < num_samples && freq_index < length)
  {
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
  }
}

__global__ void rearrangeNoise(float* input, float* output, float variance, int num_trajectories, int num_timesteps,
                               int control_dim)
{
  int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  int time_index = blockIdx.y * blockDim.y + threadIdx.y;
  int control_index = blockIdx.z * blockDim.z + threadIdx.z;
  if (sample_index < num_trajectories && time_index < num_timesteps && control_index < control_dim)
  {  // cuFFT does not normalize inverse transforms so a division by the num_timesteps is required
    output[(sample_index * num_timesteps + time_index) * control_dim + control_index] =
        input[(sample_index * control_dim + control_index) * num_timesteps + time_index] / (variance * num_timesteps);
    // printf("ROLLOUT %d CONTROL %d TIME %d: in %f out: %f\n", sample_index, control_index, time_index,
    //     input[(sample_index * control_dim + control_index) * num_timesteps + time_index],
    //     output[(sample_index * num_timesteps + time_index) * control_dim + control_index]);
  }
}

void fftfreq(const int num_samples, std::vector<float>& result, const float spacing = 1)
{
  // result is of size floor(n/2) + 1
  int result_size = num_samples / 2 + 1;
  result.clear();
  result.resize(result_size);
  for (int i = 0; i < result_size; i++)
  {
    result[i] = i / (spacing * num_samples);
  }
}

void powerlaw_psd_gaussian(float exponent, int num_timesteps, int num_trajectories, int control_dim,
                           float* control_noise_d, curandGenerator_t& gen, cudaStream_t stream = 0, float fmin = 0.0)
{
  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y = 32;
  const int BLOCKSIZE_Z = 1;
  std::vector<float> sample_freq;
  fftfreq(num_timesteps, sample_freq);
  float cutoff_freq = fmaxf(fmin, 1.0 / num_timesteps);
  int smaller_index = 0;
  for (int i = 0; i < sample_freq.size(); i++)
  {
    if (sample_freq[i] < cutoff_freq)
    {
      smaller_index++;
    }
    else if (smaller_index < sample_freq.size())
    {
      for (int j = 0; j < smaller_index; j++)
      {
        sample_freq[j] = sample_freq[smaller_index];
      }
    }
    sample_freq[i] = powf(sample_freq[i], -exponent / 2.0);
  }

  // Calculate variance
  float sigma = 0;
  std::for_each(sample_freq.begin() + 1, sample_freq.end() - 1, [&sigma](float i) { sigma += powf(i, 2); });
  sigma += powf(sample_freq.back() * ((1.0 + (num_timesteps % 2)) / 2.0), 2);
  sigma = 2 * sqrt(sigma) / num_timesteps;

  // Sample the noise in frequency domain and reutrn to time domain
  cufftHandle plan;
  const int batch = num_trajectories * control_dim;
  // Need 2 * (num_timesteps / 2 + 1) * batch of randomly sampled values
  // float* samples_in_freq_d;
  float* sigma_d;
  float* noise_in_time_d;
  cufftComplex* samples_in_freq_complex_d;
  // HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_d, sizeof(float) * 2 * batch * sample_freq.size()));
  // HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_d, sizeof(float) * 2 * batch * num_timesteps));
  HANDLE_ERROR(cudaMalloc((void**)&sigma_d, sizeof(float) * sample_freq.size()));
  HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_complex_d, sizeof(cufftComplex) * batch * sample_freq.size()));
  HANDLE_ERROR(cudaMalloc((void**)&noise_in_time_d, sizeof(float) * batch * num_timesteps));
  // curandSetStream(gen, stream);
  HANDLE_CURAND_ERROR(
      curandGenerateNormal(gen, (float*)samples_in_freq_complex_d, 2 * batch * sample_freq.size(), 0.0, 1.0));
  HANDLE_ERROR(
      cudaMemcpyAsync(sigma_d, sample_freq.data(), sizeof(float) * sample_freq.size(), cudaMemcpyHostToDevice, stream));
  const int variance_grid_x = (batch - 1) / BLOCKSIZE_X + 1;
  const int variance_grid_y = (sample_freq.size() - 1) / BLOCKSIZE_Y + 1;
  dim3 grid(variance_grid_x, variance_grid_y, 1);
  dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  // configureFrequencyNoise<<<grid, block, 0, stream>>>((cuComplex*) samples_in_freq_d, sigma_d, sample_freq.size(),
  // batch);
  configureFrequencyNoise<<<grid, block, 0, stream>>>(samples_in_freq_complex_d, sigma_d, sample_freq.size(), batch);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_CUFFT_ERROR(cufftPlan1d(&plan, num_timesteps, CUFFT_C2R, batch));
  HANDLE_CUFFT_ERROR(cufftSetStream(plan, stream));
  // freq_data needs to be batch number of num_timesteps/2 + 1 cuComplex values
  // time_data needs to be batch * num_timesteps floats
  HANDLE_CUFFT_ERROR(cufftExecC2R(plan, samples_in_freq_complex_d, noise_in_time_d));
  const int reorder_grid_x = (num_trajectories - 1) / BLOCKSIZE_X + 1;
  const int reorder_grid_y = (num_timesteps - 1) / BLOCKSIZE_Y + 1;
  const int reorder_grid_z = (control_dim - 1) / BLOCKSIZE_Z + 1;
  dim3 reorder_grid(reorder_grid_x, reorder_grid_y, reorder_grid_z);
  dim3 reorder_block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  // std::cout << "Grid: " << reorder_grid.x << ", " << reorder_grid.y << ", " << reorder_grid.z << std::endl;
  // std::cout << "Block: " << reorder_block.x << ", " << reorder_block.y << ", " << reorder_block.z << std::endl;
  rearrangeNoise<<<reorder_grid, reorder_block, 0, stream>>>(noise_in_time_d, control_noise_d, sigma, num_trajectories,
                                                             num_timesteps, control_dim);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_CUFFT_ERROR(cufftDestroy(plan));
  // HANDLE_ERROR(cudaFree(samples_in_freq_d));
  HANDLE_ERROR(cudaFree(sigma_d));
  HANDLE_ERROR(cudaFree(samples_in_freq_complex_d));
  HANDLE_ERROR(cudaFree(noise_in_time_d));
}
