#pragma once
/**
 * Created by David, Apr 11, 2021
 */

#include <curand.h>
#include <Eigen/Dense>
#include <mppi/utils/gpu_err_chk.cuh>

#include <algorithm>
#include <iostream>
#include <vector>


__global__ void createPiecewiseLinearNoise(const int num_piecewise_segments, const int num_timesteps, const int num_trajectories, const int control_dim,
      float* switch_times, float* switch_values, float* output){
  int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  int time_index = blockIdx.y * blockDim.y + threadIdx.y;
  int control_index = blockIdx.z * blockDim.z + threadIdx.z;
  if (sample_index >= num_trajectories || time_index >= num_timesteps || control_index >= control_dim){
    return;
  }

  // determine start/stop times and piecewise fraction
  int segment_index = 0;
  float start_time = 0.0f;
  float end_time = 1.0f;
  float time_frac = float(time_index) / float(num_timesteps);

  for (int i = 0; i < num_piecewise_segments; i++){
    float switch_time = switch_times[(sample_index * (num_piecewise_segments + 1) + i) * control_dim + control_index];
    if (switch_time < time_frac){
      segment_index += 1;
      if (start_time < switch_time){
        start_time = switch_time;
      }
    }
    if (switch_time > time_frac){
      if (end_time > switch_time){
        end_time = switch_time;
      }
    }  
  }

  // compute noise, interpolated between first and second value
  float first_val = switch_values[(sample_index * (num_piecewise_segments + 1) + segment_index) * control_dim + control_index];
  float second_val = switch_values[(sample_index * (num_piecewise_segments + 1) + segment_index + 1) * control_dim + control_index];
  float frac_interval = (time_frac - start_time) / (end_time - start_time);
  float output_val = (1.0f - frac_interval) * first_val + frac_interval * second_val;
  output_val = output_val * 2.0 - 1.0; // scale to [-1, 1]
  output[(sample_index * num_timesteps + time_index) * control_dim + control_index] = output_val;
  

  // print for debug
  // if (sample_index == 0){
  //   printf("sample_index: %d, time_index: %d,control_index: %d, segment_index: %d, start_time: %f, end_time: %f, time_frac: %f, first_val: %f, second_val: %f, frac_interval: %f, output: %f\n", sample_index, time_index, control_index, segment_index, start_time, end_time, time_frac, first_val, second_val, frac_interval, 
  //           output[(sample_index * num_timesteps + time_index) * control_dim + control_index]);
  // }
}

void piecewise_linear_noise(int num_piecewise_segments, int num_timesteps, int num_trajectories, int control_dim,
                           float* control_d, float* control_noise_d, float* control_std_dev_d, curandGenerator_t& gen, cudaStream_t stream = 0)
{
  // generate piecewise linear random noise, in 2 steps:
  // 1. randomly decide start/stop times of segments (uniformly divide t=0:T)
  // 2. randomly decide start and stop values of segments (uniform within bounds)

  const int BLOCKSIZE_X = 32;
  const int BLOCKSIZE_Y = 32;
  const int BLOCKSIZE_Z = 1;

  // create random start/stop times and values
  // curandSetStream(gen, stream);
  float* switch_times_d;
  float* switch_values_d;
  HANDLE_ERROR(cudaMalloc((void**)&switch_times_d, sizeof(float) * num_trajectories * (num_piecewise_segments + 1) * control_dim));
  HANDLE_ERROR(cudaMalloc((void**)&switch_values_d, sizeof(float) * num_trajectories * (num_piecewise_segments + 1) * control_dim));
  HANDLE_CURAND_ERROR(curandGenerateUniform(gen, (float*)switch_times_d, num_trajectories * (num_piecewise_segments + 1) * control_dim));
  HANDLE_CURAND_ERROR(curandGenerateUniform(gen, (float*)switch_values_d, num_trajectories * (num_piecewise_segments + 1) * control_dim));

  // create piecewise linear noise
  const int grid_x = (num_trajectories - 1) / BLOCKSIZE_X + 1;
  const int grid_y = (num_timesteps - 1) / BLOCKSIZE_Y + 1;
  const int grid_z = (control_dim - 1) / BLOCKSIZE_Z + 1;
  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  createPiecewiseLinearNoise<<<grid, block, 0, stream>>>(num_piecewise_segments, num_timesteps, num_trajectories, control_dim,
                                          switch_times_d, switch_values_d, control_noise_d);
  
  // cleanup
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaFree(switch_times_d));
  HANDLE_ERROR(cudaFree(switch_values_d));
}
