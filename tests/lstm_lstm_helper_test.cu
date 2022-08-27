#include <gtest/gtest.h>

#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <math.h>

#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>
// Auto-generated header file
#include <autorally_test_network.h>
#include <mppi/utils/network_helper_kernel_test.cuh>
#include "mppi/ddp/ddp_dynamics.h"

class LSTMLSTMHelperTest : public testing::Test
{
protected:
  void SetUp() override
  {
    generator = std::default_random_engine(7.0);
    distribution = std::normal_distribution<float>(0.0, 1.0);
  }

  void TearDown() override
  {
  }

  std::default_random_engine generator;
  std::normal_distribution<float> distribution;
};

typedef FNNHelper<FNNParams<18, 3>> FNN;
typedef FNNHelper<FNNParams<68, 100, 3>> FNN_INIT;
typedef LSTMHelper<LSTMParams<8, 10>, FNN> LSTM;
typedef LSTMHelper<LSTMParams<8, 60>, FNN_INIT> INIT_LSTM;
typedef LSTMLSTMHelper<INIT_LSTM, LSTM, 10> T;

TEST_F(LSTMLSTMHelperTest, BindStreamAndConstructor)
{
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  LSTMLSTMHelper<INIT_LSTM, LSTM, 10> helper(stream);

  EXPECT_EQ(helper.stream_, stream);
  EXPECT_EQ(helper.network_d_, nullptr);
  EXPECT_NE(helper.getLSTMModel(), nullptr);

  std::shared_ptr<INIT_LSTM> init_lstm = helper.getInitModel();
  EXPECT_NE(init_lstm, nullptr);

  auto hidden = init_lstm->getHiddenState();
  auto cell = init_lstm->getCellState();
  for (int i = 0; i < 60; i++)
  {
    EXPECT_FLOAT_EQ(hidden[i], 0.0f);
    EXPECT_FLOAT_EQ(cell[i], 0.0f);
  }
}

TEST_F(LSTMLSTMHelperTest, initializeLSTMLSTMTest)
{
  T helper;
  T::init_buffer buffer = T::init_buffer::Ones();

  auto init_params = helper.getInitParams();
  init_params.setAllValues(1.0);
  helper.setInitParams(init_params);

  std::vector<float> theta_vec(10940);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = 1.0;
  }
  helper.getInitModel()->updateOutputModel({ 68, 100, 40 }, theta_vec);

  helper.initializeLSTM(buffer);

  auto lstm = helper.getLSTMModel();
  auto hidden = lstm->getHiddenState();
  auto cell = lstm->getHiddenState();
  for (int i = 0; i < T::HIDDEN_DIM; i++)
  {
    EXPECT_FLOAT_EQ(hidden[i], 101.0f);
    EXPECT_FLOAT_EQ(cell[i], 101.0f);
  }
}

TEST_F(LSTMLSTMHelperTest, GPUSetupAndParamsCheck)
{
  T model;

  std::vector<float> theta_vec(T::OUTPUT_PARAMS_T::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = distribution(generator);
  }
  model.updateOutputModel({ 18, 3 }, theta_vec);
  auto params = model.getLSTMParams();
  for (int i = 0; i < 10 * 10; i++)
  {
    params.W_im[i] = distribution(generator);
    params.W_fm[i] = distribution(generator);
    params.W_om[i] = distribution(generator);
    params.W_cm[i] = distribution(generator);
  }
  for (int i = 0; i < 8 * 10; i++)
  {
    params.W_ii[i] = distribution(generator);
    params.W_fi[i] = distribution(generator);
    params.W_oi[i] = distribution(generator);
    params.W_ci[i] = distribution(generator);
  }
  for (int i = 0; i < 10; i++)
  {
    params.b_i[i] = distribution(generator);
    params.b_f[i] = distribution(generator);
    params.b_o[i] = distribution(generator);
    params.b_c[i] = distribution(generator);
    params.initial_hidden[i] = distribution(generator);
    params.initial_cell[i] = distribution(generator);
  }
  model.setLSTMParams(params);

  int grid_dim = 5;

  std::vector<T::LSTM_PARAMS_T> lstm_params(grid_dim);
  std::vector<T::LSTM_PARAMS_T> shared_lstm_params(grid_dim);
  std::vector<T::OUTPUT_PARAMS_T> fnn_params(grid_dim);
  std::vector<T::OUTPUT_PARAMS_T> shared_fnn_params(grid_dim);

  EXPECT_EQ(model.GPUMemStatus_, false);
  EXPECT_EQ(model.network_d_, nullptr);
  EXPECT_EQ(model.getLSTMModel()->GPUMemStatus_, false);
  EXPECT_EQ(model.getLSTMModel()->network_d_, nullptr);
  EXPECT_NE(model.getLSTMModel()->getOutputModel(), nullptr);
  EXPECT_EQ(model.getInitModel()->GPUMemStatus_, false);
  EXPECT_EQ(model.getInitModel()->network_d_, nullptr);
  EXPECT_NE(model.getInitModel()->getOutputModel(), nullptr);

  model.GPUSetup();

  EXPECT_EQ(model.GPUMemStatus_, true);
  EXPECT_NE(model.network_d_, nullptr);
  EXPECT_EQ(model.getLSTMModel()->GPUMemStatus_, true);
  EXPECT_NE(model.getLSTMModel()->network_d_, nullptr);
  EXPECT_NE(model.getLSTMModel()->getOutputModel(), nullptr);
  EXPECT_EQ(model.getInitModel()->GPUMemStatus_, false);
  EXPECT_EQ(model.getInitModel()->network_d_, nullptr);
  EXPECT_NE(model.getInitModel()->getOutputModel(), nullptr);

  // launch kernel
  launchParameterCheckTestKernel<T>(model, lstm_params, shared_lstm_params, fnn_params, shared_fnn_params);

  for (int grid = 0; grid < grid_dim; grid++)
  {
    // ensure that the output nn matches
    for (int i = 0; i < theta_vec.size(); i++)
    {
      EXPECT_FLOAT_EQ(fnn_params[grid].theta[i], theta_vec[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_fnn_params[grid].theta[i], theta_vec[i]) << "at index " << i;
    }
    EXPECT_EQ(fnn_params[grid].stride_idcs[0], 0);
    EXPECT_EQ(fnn_params[grid].stride_idcs[1], 54);
    EXPECT_EQ(shared_fnn_params[grid].stride_idcs[0], 0) << "at grid " << grid;
    EXPECT_EQ(shared_fnn_params[grid].stride_idcs[1], 54) << "at grid " << grid;

    EXPECT_EQ(fnn_params[grid].net_structure[0], 18);
    EXPECT_EQ(fnn_params[grid].net_structure[1], 3);
    EXPECT_EQ(shared_fnn_params[grid].net_structure[0], 18) << "at grid " << grid;
    EXPECT_EQ(shared_fnn_params[grid].net_structure[1], 3) << "at grid " << grid;

    for (int i = 0; i < 10 * 10; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].W_im[i], params.W_im[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_fm[i], params.W_fm[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_cm[i], params.W_cm[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_om[i], params.W_om[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_im[i], params.W_im[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_fm[i], params.W_fm[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_cm[i], params.W_cm[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_om[i], params.W_om[i]);
    }

    for (int i = 0; i < 8 * 10; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].W_ii[i], params.W_ii[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_fi[i], params.W_fi[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_oi[i], params.W_oi[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_ci[i], params.W_ci[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_ii[i], params.W_ii[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_fi[i], params.W_fi[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_oi[i], params.W_oi[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_ci[i], params.W_ci[i]);
    }

    for (int i = 0; i < 10; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].b_i[i], params.b_i[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].b_f[i], params.b_f[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].b_o[i], params.b_o[i]);
      EXPECT_FLOAT_EQ(lstm_params[grid].b_c[i], params.b_c[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_i[i], params.b_i[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_f[i], params.b_f[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_o[i], params.b_o[i]);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_c[i], params.b_c[i]);
    }

    for (int i = 0; i < 10; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].initial_hidden[i], params.initial_hidden[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(lstm_params[grid].initial_cell[i], params.initial_cell[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].initial_hidden[i], params.initial_hidden[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].initial_cell[i], params.initial_cell[i]) << "at index " << i;
    }
  }
}

TEST_F(LSTMLSTMHelperTest, forwardCPU)
{
  T model;

  std::vector<float> theta_vec(LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = 1.0;
  }
  model.updateOutputModel({ 18, 3 }, theta_vec);

  theta_vec.resize(T::INIT_OUTPUT_PARAMS_T::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = 1.0;
  }
  model.updateOutputModelInit({ 68, 100, 40 }, theta_vec);

  auto lstm_params = model.getLSTMParams();
  lstm_params.setAllValues(1.0f);
  model.setLSTMParams(lstm_params);

  auto init_params = model.getInitParams();
  init_params.setAllValues(0.2f);
  model.setInitParams(init_params);

  for (int i = 0; i < T::HIDDEN_DIM; i++)
  {
    EXPECT_FLOAT_EQ(model.getInitModel()->getHiddenState()[i], 0.2);
    EXPECT_FLOAT_EQ(model.getInitModel()->getCellState()[i], 0.2);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getHiddenState()[i], 1.0);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getCellState()[i], 1.0);
  }

  LSTM::input_array input = LSTM::input_array::Ones() * 0.3;
  LSTM::output_array output;

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 28.28055);
  EXPECT_FLOAT_EQ(output[1], 28.28055);
  EXPECT_FLOAT_EQ(output[2], 28.28055);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 28.901096);
  EXPECT_FLOAT_EQ(output[1], 28.901096);
  EXPECT_FLOAT_EQ(output[2], 28.901096);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 28.986588);
  EXPECT_FLOAT_EQ(output[1], 28.986588);
  EXPECT_FLOAT_EQ(output[2], 28.986588);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 28.998184);
  EXPECT_FLOAT_EQ(output[1], 28.998184);
  EXPECT_FLOAT_EQ(output[2], 28.998184);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 28.999764);
  EXPECT_FLOAT_EQ(output[1], 28.999764);
  EXPECT_FLOAT_EQ(output[2], 28.999764);

  T::init_buffer buffer = T::init_buffer::Ones();
  model.initializeLSTM(buffer);

  for (int i = 0; i < T::HIDDEN_DIM; i++)
  {
    EXPECT_FLOAT_EQ(model.getInitModel()->getHiddenState()[i], 1.0);
    EXPECT_FLOAT_EQ(model.getInitModel()->getCellState()[i], 11.0);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getHiddenState()[i], 101.0);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getCellState()[i], 101.0);
  }

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 29);
  EXPECT_FLOAT_EQ(output[1], 29);
  EXPECT_FLOAT_EQ(output[2], 29);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 29);
  EXPECT_FLOAT_EQ(output[1], 29);
  EXPECT_FLOAT_EQ(output[2], 29);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 29);
  EXPECT_FLOAT_EQ(output[1], 29);
  EXPECT_FLOAT_EQ(output[2], 29);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 29);
  EXPECT_FLOAT_EQ(output[1], 29);
  EXPECT_FLOAT_EQ(output[2], 29);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 29);
  EXPECT_FLOAT_EQ(output[1], 29);
  EXPECT_FLOAT_EQ(output[2], 29);
}

// TEST_F(LSTMLSTMHelperTest, forwardGPU)
// {
//   const int num_rollouts = 1;
//
//   T model;
//
//   std::vector<float> theta_vec(LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
//   for (int i = 0; i < theta_vec.size(); i++)
//   {
//     theta_vec[i] = 1.0;
//   }
//   model.updateOutputModel({ 18, 3 }, theta_vec);
//
//   theta_vec.resize(T::INIT_OUTPUT_PARAMS_T::NUM_PARAMS);
//   for (int i = 0; i < theta_vec.size(); i++)
//   {
//     theta_vec[i] = 1.0;
//   }
//   model.updateOutputModelInit({ 68, 100, 40 }, theta_vec);
//
//   auto lstm_params = model.getLSTMParams();
//   lstm_params.setAllValues(1.0f);
//   model.setLSTMParams(lstm_params);
//
//   auto init_params = model.getInitParams();
//   init_params.setAllValues(0.2f);
//   model.setInitParams(init_params);
//
//   model.GPUSetup();
//
//   std::vector<float> theta(T::OUTPUT_PARAMS_T::NUM_PARAMS);
//   std::fill(theta.begin(), theta.end(), 1);
//   model.updateOutputModel({ 18, 3 }, theta);
//
//   auto params = model.getLSTMParams();
//   for (int i = 0; i < 10 * 10; i++)
//   {
//     params.W_im[i] = 1.0;
//     params.W_fm[i] = 1.0;
//     params.W_om[i] = 1.0;
//     params.W_cm[i] = 1.0;
//   }
//   for (int i = 0; i < 8 * 10; i++)
//   {
//     params.W_ii[i] = 1.0;
//     params.W_fi[i] = 1.0;
//     params.W_oi[i] = 1.0;
//     params.W_ci[i] = 1.0;
//   }
//   for (int i = 0; i < 10; i++)
//   {
//     params.b_i[i] = 1.0;
//     params.b_f[i] = 1.0;
//     params.b_o[i] = 1.0;
//     params.b_c[i] = 1.0;
//     params.initial_hidden[i] = 1.0;
//     params.initial_cell[i] = 1.0;
//   }
//   model.setLSTMParams(params);
//
//   Eigen::Matrix<float, T::INPUT_DIM, num_rollouts> inputs;
//   inputs = Eigen::Matrix<float, T::INPUT_DIM, num_rollouts>::Ones();
//   T::output_array output;
//
//   std::array<float, 5> true_vals = { 29, 29, 29, 29, 29 };
//
//   std::vector<std::array<float, T::INPUT_DIM>> input_arr(num_rollouts);
//   std::vector<std::array<float, T::OUTPUT_DIM>> output_arr(num_rollouts);
//
//   T::init_buffer buffer = T::init_buffer::Ones();
//   model.initializeLSTM(buffer);
//
//   for (int y_dim = 1; y_dim < 16; y_dim++)
//   {
//     for (int state_index = 0; state_index < num_rollouts; state_index++)
//     {
//       for (int dim = 0; dim < input_arr[0].size(); dim++)
//       {
//         input_arr[state_index][dim] = inputs.col(state_index)(dim);
//       }
//     }
//
//     for (int step = 1; step < 6; step++)
//     {
//       launchForwardTestKernel<T, 32>(model, input_arr, output_arr, y_dim, step);
//       for (int point = 0; point < num_rollouts; point++)
//       {
//         model.initializeLSTM(buffer);
//         T::input_array input = inputs.col(point);
//         T::output_array output;
//
//         for (int cpu_step = 0; cpu_step < step; cpu_step++)
//         {
//           model.forward(input, output);
//         }
//         for (int dim = 0; dim < T::INPUT_DIM; dim++)
//         {
//           EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//         }
//         for (int dim = 0; dim < T::OUTPUT_DIM; dim++)
//         {
//           EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//           EXPECT_TRUE(isfinite(output_arr[point][dim]));
//           EXPECT_FLOAT_EQ(output(dim), true_vals[step - 1]) << "at dim " << dim << " step " << step;
//         }
//       }
//     }
//   }
// }

// using LSTM2 = LSTMLSTMHelper<LSTMParams<8, 20, 0>, FNNHelper<FNNParams<28, 3>>>;
// TEST_F(LSTMLSTMHelperTest, forwardGPU)
// {
//   const int num_rollouts = 1;
//
//   LSTM2 model;
//   model.GPUSetup();
//
//   std::vector<float> theta(87);
//   std::fill(theta.begin(), theta.end(), 1);
//   model.updateOutputModel({ 28, 3 }, theta);
//
//   auto params = model.getLSTMParams();
//   for (int i = 0; i < 20 * 20; i++)
//   {
//     params.W_im[i] = 1.0;
//     params.W_fm[i] = 1.0;
//     params.W_om[i] = 1.0;
//     params.W_cm[i] = 1.0;
//   }
//   for (int i = 0; i < 8 * 20; i++)
//   {
//     params.W_ii[i] = 1.0;
//     params.W_fi[i] = 1.0;
//     params.W_oi[i] = 1.0;
//     params.W_ci[i] = 1.0;
//   }
//   for (int i = 0; i < 20; i++)
//   {
//     params.b_i[i] = 1.0;
//     params.b_f[i] = 1.0;
//     params.b_o[i] = 1.0;
//     params.b_c[i] = 1.0;
//     params.initial_hidden[i] = 1.0;
//     params.initial_cell[i] = 1.0;
//   }
//   model.updateLSTM(params);
//
//   Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts> inputs;
//   inputs = Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts>::Ones();
//   LSTM2::output_array output;
//
//   std::array<float, 5> true_vals = { 28.28055, 28.901096, 28.986588, 28.998184, 28.999756 };
//
//   std::vector<std::array<float, LSTM2::INPUT_DIM>> input_arr(num_rollouts);
//   std::vector<std::array<float, 3>> output_arr(num_rollouts);
//
//   for (int y_dim = 1; y_dim < 16; y_dim++)
//   {
//     for (int state_index = 0; state_index < num_rollouts; state_index++)
//     {
//       for (int dim = 0; dim < input_arr[0].size(); dim++)
//       {
//         input_arr[state_index][dim] = inputs.col(state_index)(dim);
//       }
//     }
//
//     for (int step = 1; step < 6; step++)
//     {
//       launchForwardTestKernel<LSTM2, 32>(model, input_arr, output_arr, y_dim, step);
//       for (int point = 0; point < num_rollouts; point++)
//       {
//         model.resetInitialStateCPU();
//         LSTM2::input_array input = inputs.col(point);
//         LSTM2::output_array output;
//
//         for (int cpu_step = 0; cpu_step < step; cpu_step++)
//         {
//           model.forward(input, output);
//         }
//         for (int dim = 0; dim < LSTM2::INPUT_DIM; dim++)
//         {
//           EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//         }
//         for (int dim = 0; dim < LSTM2::OUTPUT_DIM; dim++)
//         {
//           EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//           EXPECT_TRUE(isfinite(output_arr[point][dim]));
//           EXPECT_FLOAT_EQ(output(dim), true_vals[step - 1]) << "at dim " << dim << " step " << step;
//         }
//       }
//     }
//   }
// }
//
// TEST_F(LSTMLSTMHelperTest, forwardGPUCompare)
// {
//   const int num_rollouts = 1000;
//
//   LSTM2 model;
//   model.GPUSetup();
//
//   std::vector<float> theta(87);
//   std::fill(theta.begin(), theta.end(), 1);
//   model.updateOutputModel({ 28, 3 }, theta);
//
//   auto params = model.getLSTMParams();
//   for (int i = 0; i < 20 * 20; i++)
//   {
//     params.W_im[i] = 1.0;
//     params.W_fm[i] = 1.0;
//     params.W_om[i] = 1.0;
//     params.W_cm[i] = 1.0;
//   }
//   for (int i = 0; i < 8 * 20; i++)
//   {
//     params.W_ii[i] = 1.0;
//     params.W_fi[i] = 1.0;
//     params.W_oi[i] = 1.0;
//     params.W_ci[i] = 1.0;
//   }
//   for (int i = 0; i < 20; i++)
//   {
//     params.b_i[i] = 1.0;
//     params.b_f[i] = 1.0;
//     params.b_o[i] = 1.0;
//     params.b_c[i] = 1.0;
//     params.initial_hidden[i] = 1.0;
//     params.initial_cell[i] = 1.0;
//   }
//   model.updateLSTM(params);
//
//   Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts> inputs;
//   inputs = Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts>::Random();
//   LSTM2::output_array output;
//
//   std::vector<std::array<float, LSTM2::INPUT_DIM>> input_arr(num_rollouts);
//   std::vector<std::array<float, 3>> output_arr(num_rollouts);
//
//   for (int y_dim = 1; y_dim < 16; y_dim++)
//   {
//     for (int state_index = 0; state_index < num_rollouts; state_index++)
//     {
//       for (int dim = 0; dim < input_arr[0].size(); dim++)
//       {
//         input_arr[state_index][dim] = inputs.col(state_index)(dim);
//       }
//     }
//     for (int step = 1; step < 6; step++)
//     {
//       launchForwardTestKernel<LSTM2, 32>(model, input_arr, output_arr, y_dim, step);
//       for (int point = 0; point < num_rollouts; point++)
//       {
//         model.resetInitialStateCPU();
//         LSTM2::input_array input = inputs.col(point);
//         LSTM2::output_array output;
//
//         for (int cpu_step = 0; cpu_step < step; cpu_step++)
//         {
//           model.forward(input, output);
//         }
//         for (int dim = 0; dim < LSTM2::INPUT_DIM; dim++)
//         {
//           EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//         }
//         for (int dim = 0; dim < LSTM2::OUTPUT_DIM; dim++)
//         {
//           EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//           EXPECT_TRUE(isfinite(output_arr[point][dim]));
//         }
//       }
//     }
//   }
// }

// TEST_F(LSTMLSTMHelperTest, TestComputeGradComputationFinite)
// {
//   LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>> model;
//   std::vector<float> theta(1412);
//   for (int i = 0; i < 1412; i++)
//   {
//     theta[i] = distribution(generator);
//   }
//   model.updateModel({ 6, 32, 32, 4 }, theta);
//
//   LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx numeric_jac;
//   LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx analytic_jac;
//
//   for (int i = 0; i < 1000; i++)
//   {
//     LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>>::input_array input;
//     input = LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>>::input_array::Random();
//
//     model.computeGrad(input, analytic_jac);
//     EXPECT_TRUE(analytic_jac.allFinite());
//   }
// }
//
// TEST_F(LSTMLSTMHelperTest, TestComputeGradComputationCompare)
// {
//   GTEST_SKIP();
//   LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>> model;
//   std::vector<float> theta(1412);
//   for (int i = 0; i < 1412; i++)
//   {
//     theta[i] = distribution(generator);
//   }
//   model.updateModel({ 6, 32, 32, 4 }, theta);
//
//   LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx numeric_jac;
//   LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx analytic_jac;
//
//   LSTMLSTMHelper<LSTMParams<6, 32, 32, 4>>::input_array input;
//   input << 1, 2, 3, 4, 5, 6;
//
//   model.computeGrad(input, analytic_jac);
//
//   // numeric_jac = num_diff.df(input, numeric_jac);
//
//   ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n"
//                                                        << numeric_jac << "\nAnalytic Jacobian\n"
//                                                        << analytic_jac;
// }
