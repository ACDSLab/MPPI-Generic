//
// Created by Bogdan on 12/26/21
//

#include <gtest/gtest.h>
#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh>

#include <numeric>

void assert_float_rel_near(const float known, const float compute, float rel_err)
{
  float err = fabsf(compute - known) / known;
  ASSERT_NEAR(known, compute, rel_err) << "Relative error is " << err;
}

TEST(cuFFT, checkErrorCode)
{
  cufftHandle plan;
  cuComplex* input_d;
  float* output_d;
  // As this call is intended to cause issues, disable compiler warning
  // src: https://stackoverflow.com/questions/14831051/how-to-disable-a-specific-nvcc-compiler-warnings
  // https://stackoverflow.com/questions/56193080/how-do-i-apply-a-flag-setting-nvcc-pragma-to-only-a-few-lines-of-code
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#nv-diagnostic-pragmas
#pragma push
#pragma diag_suppress = used_before_set
  auto status = cufftExecC2R(plan, input_d, output_d);
#pragma pop
  // cufftAssert(status, __FILE__, __LINE__);
  std::string error_string = cufftGetErrorString(status);
  // std::cout << error_string << std::endl;
  EXPECT_TRUE(error_string == "cuFFT was passed an invalid plan handle");
}

TEST(ColoredNoise, checkWhiteNoise)
{
  int NUM_TIMESTEPS = 50000;
  int NUM_ROLLOUTS = 1;
  int CONTROL_DIM = 1;
  std::vector<float> exponents(CONTROL_DIM, 0.0);
  int full_buffer_size = NUM_ROLLOUTS * NUM_TIMESTEPS * CONTROL_DIM;
  float* colored_noise_d;
  float colored_noise_output[full_buffer_size] = { 0 };
  HANDLE_ERROR(cudaMalloc((void**)&colored_noise_d, sizeof(float) * full_buffer_size));
  cudaStream_t stream;
  curandGenerator_t gen;
  cudaStreamCreate(&stream);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandSetStream(gen, stream);

  powerlaw_psd_gaussian(exponents, NUM_TIMESTEPS, NUM_ROLLOUTS, colored_noise_d, gen, stream);
  HANDLE_ERROR(cudaMemcpyAsync(colored_noise_output, colored_noise_d, sizeof(float) * full_buffer_size,
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  // Check percentages for 3 standard deviations
  std::vector<int> num_within_std_dev(3, 0);
  for (int i = 0; i < full_buffer_size; i++)
  {
    for (int j = 0; j < num_within_std_dev.size(); j++)
    {
      if (fabsf(colored_noise_output[i]) < j + 1.0)
      {
        num_within_std_dev[j]++;
        break;
      }
    }
  }

  float perc_within_n_std_dev[num_within_std_dev.size()];
  // Percentages from https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
  float known_percentages[3] = { 0.6827, 0.9545, 0.9973 };
  for (int i = 0; i < num_within_std_dev.size(); i++)
  {
    perc_within_n_std_dev[i] =
        std::accumulate(num_within_std_dev.begin(), num_within_std_dev.begin() + i + 1, 0.0) / full_buffer_size;
    assert_float_rel_near(known_percentages[i], perc_within_n_std_dev[i], 0.001);
  }
}

TEST(ColoredNoise, checkPinkNoise)
{
  int NUM_TIMESTEPS = 50000;
  int NUM_ROLLOUTS = 1;
  int CONTROL_DIM = 1;
  std::vector<float> exponents(CONTROL_DIM, 1.0);
  int full_buffer_size = NUM_ROLLOUTS * NUM_TIMESTEPS * CONTROL_DIM;
  float* colored_noise_d;
  float colored_noise_output[full_buffer_size] = { 0 };
  HANDLE_ERROR(cudaMalloc((void**)&colored_noise_d, sizeof(float) * full_buffer_size));
  cudaStream_t stream;
  curandGenerator_t gen;
  cudaStreamCreate(&stream);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandSetStream(gen, stream);

  powerlaw_psd_gaussian(exponents, NUM_TIMESTEPS, NUM_ROLLOUTS, colored_noise_d, gen, stream);
  HANDLE_ERROR(cudaMemcpyAsync(colored_noise_output, colored_noise_d, sizeof(float) * full_buffer_size,
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  int within_std_dev = 0;
  int within_2_std_dev = 0;
  for (int i = 0; i < full_buffer_size; i++)
  {
    if (fabsf(colored_noise_output[i]) < 1.0)
    {
      within_std_dev++;
    }
    else if (fabsf(colored_noise_output[i]) < 2.0)
    {
      within_2_std_dev++;
    }
  }
  float perc_one_std_dev = (float)within_std_dev / full_buffer_size;
  float perc_two_std_dev = (float)(within_std_dev + within_2_std_dev) / full_buffer_size;
  std::cout << "Percentage within 1 std dev: " << 100 * perc_one_std_dev << std::endl;
  std::cout << "Percentage within 2 std dev: " << 100 * perc_two_std_dev << std::endl;
  // assert_float_rel_near(0.6827, perc_one_std_dev, 0.001);
  // assert_float_rel_near(0.9545, perc_two_std_dev, 0.001);
}

TEST(ColoredNoise, checkRedNoise)
{
  int NUM_TIMESTEPS = 50000;
  int NUM_ROLLOUTS = 1;
  int CONTROL_DIM = 1;
  std::vector<float> exponents(CONTROL_DIM, 2.0);
  int full_buffer_size = NUM_ROLLOUTS * NUM_TIMESTEPS * CONTROL_DIM;
  float* colored_noise_d;
  float colored_noise_output[full_buffer_size] = { 0 };
  HANDLE_ERROR(cudaMalloc((void**)&colored_noise_d, sizeof(float) * full_buffer_size));
  cudaStream_t stream;
  curandGenerator_t gen;
  cudaStreamCreate(&stream);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandSetStream(gen, stream);

  powerlaw_psd_gaussian(exponents, NUM_TIMESTEPS, NUM_ROLLOUTS, colored_noise_d, gen, stream);
  HANDLE_ERROR(cudaMemcpyAsync(colored_noise_output, colored_noise_d, sizeof(float) * full_buffer_size,
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  // Check percentages for 3 standard deviations
  std::vector<int> num_within_std_dev(3, 0);
  for (int i = 0; i < full_buffer_size; i++)
  {
    for (int j = 0; j < num_within_std_dev.size(); j++)
    {
      if (fabsf(colored_noise_output[i]) < j + 1.0)
      {
        num_within_std_dev[j]++;
        break;
      }
    }
  }

  float perc_within_n_std_dev[num_within_std_dev.size()];
  // Percentages from https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
  float known_percentages[3] = { 0.6827, 0.9545, 0.9973 };
  for (int i = 0; i < num_within_std_dev.size(); i++)
  {
    perc_within_n_std_dev[i] =
        std::accumulate(num_within_std_dev.begin(), num_within_std_dev.begin() + i + 1, 0.0) / full_buffer_size;
    std::cout << "Percentage within " << i + 1 << " std dev: " << 100 * perc_within_n_std_dev[i] << std::endl;
    // assert_float_rel_near(known_percentages[i], perc_within_n_std_dev[i], 0.001);
  }
}

TEST(ColoredNoise, checkMultiNoise)
{
  int NUM_TIMESTEPS = 6000;
  int NUM_ROLLOUTS = 50;
  int CONTROL_DIM = 3;
  std::vector<float> exponents(CONTROL_DIM, 0.0);
  exponents[1] = 0.5;
  exponents[2] = 2.0;
  // exponents[3] = 1.25;
  // exponents[4] = 0.75;
  int full_buffer_size = NUM_ROLLOUTS * NUM_TIMESTEPS * CONTROL_DIM;
  float* colored_noise_d;
  float colored_noise_output[full_buffer_size] = { 0 };
  HANDLE_ERROR(cudaMalloc((void**)&colored_noise_d, sizeof(float) * full_buffer_size));
  cudaStream_t stream;
  curandGenerator_t gen;
  cudaStreamCreate(&stream);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandSetStream(gen, stream);

  powerlaw_psd_gaussian(exponents, NUM_TIMESTEPS, NUM_ROLLOUTS, colored_noise_d, gen, stream);
  HANDLE_ERROR(cudaMemcpyAsync(colored_noise_output, colored_noise_d, sizeof(float) * full_buffer_size,
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  // Check percentages for 3 standard deviations
  std::vector<int> num_within_std_dev(3, 0);
  float perc_within_n_std_dev[num_within_std_dev.size()];
  // Percentages from https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
  float known_percentages[3] = { 0.6827, 0.9545, 0.9973 };
  for (int c = 0; c < CONTROL_DIM; c++)
  {
    std::fill(num_within_std_dev.begin(), num_within_std_dev.end(), 0);
    for (int i = 0; i < NUM_TIMESTEPS * NUM_ROLLOUTS; i++)
    {
      for (int j = 0; j < num_within_std_dev.size(); j++)
      {
        if (fabsf(colored_noise_output[i * CONTROL_DIM + c]) < j + 1.0)
        {
          num_within_std_dev[j]++;
          break;
        }
      }
    }

    for (int i = 0; i < num_within_std_dev.size(); i++)
    {
      perc_within_n_std_dev[i] = std::accumulate(num_within_std_dev.begin(), num_within_std_dev.begin() + i + 1, 0.0) /
                                 (NUM_ROLLOUTS * NUM_TIMESTEPS);
      std::cout << "Colored Noise " << exponents[c] << " ";
      std::cout << "percent of samples within " << i + 1 << " std dev: " << 100 * perc_within_n_std_dev[i] << std::endl;
      if (exponents[c] == 0)
      {
        assert_float_rel_near(known_percentages[i], perc_within_n_std_dev[i], 0.001);
      }
    }
  }
}
