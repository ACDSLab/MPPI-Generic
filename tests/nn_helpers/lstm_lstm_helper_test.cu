// #include <gtest/gtest.h>

#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <math.h>

#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>
// Auto-generated header file
#include <test_networks.h>
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

template class FNNParams<18, 3>;
template class FNNParams<68, 100, 20>;
template class LSTMParams<8, 10>;
template class LSTMParams<8, 60>;
typedef FNNParams<18, 3> FNN_PARAMS;
typedef FNNParams<68, 100, 20> FNN_INIT_PARAMS;
typedef LSTMHelper<LSTMParams<8, 10>, FNN_PARAMS> LSTM;
typedef LSTMHelper<LSTMParams<8, 60>, FNN_INIT_PARAMS> INIT_LSTM;

template class LSTMLSTMHelper<INIT_LSTM, LSTM, 10>;

typedef LSTMLSTMHelper<INIT_LSTM, LSTM, 10> T;

TEST_F(LSTMLSTMHelperTest, BindStreamAndConstructor)
{
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  LSTMLSTMHelper<INIT_LSTM, LSTM, 10> helper(stream);

  EXPECT_EQ(helper.getLSTMModel()->stream_, stream);
  EXPECT_NE(helper.getLSTMModel(), nullptr);

  auto init_lstm = helper.getInitModel();
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

  auto init_params = helper.getInitLSTMParams();
  init_params.setAllValues(1.0);
  helper.setInitParams(init_params);

  std::vector<float> theta_vec(FNN_INIT_PARAMS::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = 1.0;
  }
  helper.getInitModel()->updateOutputModel({ 68, 100, 20 }, theta_vec);

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

  EXPECT_EQ(model.getLSTMModel()->GPUMemStatus_, false);
  EXPECT_EQ(model.getLSTMModel()->network_d_, nullptr);
  EXPECT_NE(model.getLSTMModel()->getOutputModel(), nullptr);
  EXPECT_EQ(model.getInitModel()->GPUMemStatus_, false);
  EXPECT_EQ(model.getInitModel()->network_d_, nullptr);
  EXPECT_NE(model.getInitModel()->getOutputModel(), nullptr);

  model.GPUSetup();

  EXPECT_EQ(model.getLSTMModel()->GPUMemStatus_, true);
  EXPECT_NE(model.getLSTMModel()->network_d_, nullptr);
  EXPECT_NE(model.getLSTMModel()->getOutputModel(), nullptr);
  EXPECT_EQ(model.getInitModel()->GPUMemStatus_, false);
  EXPECT_EQ(model.getInitModel()->network_d_, nullptr);
  EXPECT_NE(model.getInitModel()->getOutputModel(), nullptr);

  // launch kernel
  launchParameterCheckTestKernel<LSTM>(*model.getLSTMModel(), lstm_params, shared_lstm_params, fnn_params,
                                       shared_fnn_params);

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

TEST_F(LSTMLSTMHelperTest, LoadModelPathTest)
{
  using LSTM = LSTMHelper<LSTMParams<3, 25>, FNNParams<28, 30, 30, 2>>;
  using INIT_LSTM = LSTMHelper<LSTMParams<3, 60>, FNNParams<63, 15, 15, 50>>;
  using NN = LSTMLSTMHelper<INIT_LSTM, LSTM, 6>;

  NN model;

  int num_points = 1;
  int T = 1;

  std::string path = mppi::tests::test_lstm_lstm;
  if (!fileExists(path))
  {
    std::cerr << "Could not load neural net model at path: " << path.c_str();
    exit(-1);
  }
  model.loadParams("", path);

  cnpy::npz_t input_outputs = cnpy::npz_load(path);
  double* inputs = input_outputs.at("input").data<double>();
  double* outputs = input_outputs.at("output").data<double>();
  double* init_inputs = input_outputs.at("init_input").data<double>();
  double* init_hidden = input_outputs.at("init_hidden").data<double>();
  double* init_cell = input_outputs.at("init_cell").data<double>();
  double* hidden = input_outputs.at("hidden").data<double>();
  double* cell = input_outputs.at("cell").data<double>();

  double tol = 1e-5;

  LSTM::input_array input;
  LSTM::output_array output;

  for (int point = 0; point < num_points; point++)
  {
    // run the init network and ensure initial hidden/cell match
    NN::init_buffer buffer;
    model.resetInitHiddenCPU();
    for (int t = 0; t < 6; t++)
    {
      for (int i = 0; i < 3; i++)
      {
        buffer(i, t) = init_inputs[point * 6 * 3 + t * 3 + i];
      }
    }
    model.initializeLSTM(buffer);

    for (int i = 0; i < 25; i++)
    {
      EXPECT_NEAR(model.getLSTMModel()->getHiddenState()(i), init_hidden[25 * point + i], tol)
          << "at point " << point << " index " << i;
      EXPECT_NEAR(model.getLSTMModel()->getCellState()(i), init_cell[25 * point + i], tol);
    }
    for (int t = 0; t < T; t++)
    {
      for (int i = 0; i < 3; i++)
      {
        input(i) = inputs[point * T * 3 + t * 3 + i];
      }
      model.forward(input, output);

      for (int i = 0; i < 2; i++)
      {
        EXPECT_NEAR(output[i], outputs[point * T * 2 + t * 2 + i], tol) << "point " << point << " at dim " << i;
      }
      for (int i = 0; i < 25; i++)
      {
        EXPECT_NEAR(model.getLSTMModel()->getHiddenState()[i], hidden[point * T * 25 + 25 * t + i], tol)
            << "point " << point << " at dim " << i;
        EXPECT_NEAR(model.getLSTMModel()->getCellState()[i], cell[point * T * 25 + 25 * t + i], tol)
            << "point " << point << " at dim " << i;
      }
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
    theta_vec[i] = 0.01;
  }
  model.updateOutputModelInit({ 68, 100, 20 }, theta_vec);

  auto lstm_params = model.getLSTMParams();
  lstm_params.setAllValues(1.0f);
  model.setLSTMParams(lstm_params);

  auto init_params = model.getInitLSTMParams();
  init_params.setAllValues(0.1f);
  model.setInitParams(init_params);

  for (int i = 0; i < T::HIDDEN_DIM; i++)
  {
    EXPECT_FLOAT_EQ(model.getInitModel()->getHiddenState()[i], 0.1);
    EXPECT_FLOAT_EQ(model.getInitModel()->getCellState()[i], 0.1);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getHiddenState()[i], 1.0);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getCellState()[i], 1.0);
  }

  LSTM::input_array input = LSTM::input_array::Ones();
  LSTM::output_array output;

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 18.640274);
  EXPECT_FLOAT_EQ(output[1], 18.640274);
  EXPECT_FLOAT_EQ(output[2], 18.640274);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 18.950548);
  EXPECT_FLOAT_EQ(output[1], 18.950548);
  EXPECT_FLOAT_EQ(output[2], 18.950548);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 18.993294);
  EXPECT_FLOAT_EQ(output[1], 18.993294);
  EXPECT_FLOAT_EQ(output[2], 18.993294);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 18.999092);
  EXPECT_FLOAT_EQ(output[1], 18.999092);
  EXPECT_FLOAT_EQ(output[2], 18.999092);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 18.999878);
  EXPECT_FLOAT_EQ(output[1], 18.999878);
  EXPECT_FLOAT_EQ(output[2], 18.999878);

  T::init_buffer buffer = T::init_buffer::Ones() * 0.1;
  model.initializeLSTM(buffer);

  for (int i = 0; i < T::HIDDEN_DIM; i++)
  {
    EXPECT_FLOAT_EQ(model.getInitModel()->getHiddenState()[i], 0.99790788);
    EXPECT_FLOAT_EQ(model.getInitModel()->getCellState()[i], 9.2190571);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getHiddenState()[i], 0.558857381);
    EXPECT_FLOAT_EQ(model.getLSTMModel()->getCellState()[i], 0.558857381);
  }

  input *= 0.1;

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 10.945133);
  EXPECT_FLOAT_EQ(output[1], 10.945133);
  EXPECT_FLOAT_EQ(output[2], 10.945133);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 11.680509);
  EXPECT_FLOAT_EQ(output[1], 11.680509);
  EXPECT_FLOAT_EQ(output[2], 11.680509);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 11.783686);
  EXPECT_FLOAT_EQ(output[1], 11.783686);
  EXPECT_FLOAT_EQ(output[2], 11.783686);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 11.797728);
  EXPECT_FLOAT_EQ(output[1], 11.797728);
  EXPECT_FLOAT_EQ(output[2], 11.797728);

  model.forward(input, output);
  EXPECT_FLOAT_EQ(output[0], 11.79963);
  EXPECT_FLOAT_EQ(output[1], 11.79963);
  EXPECT_FLOAT_EQ(output[2], 11.79963);
}

TEST_F(LSTMLSTMHelperTest, forwardGPU)
{
  const int num_rollouts = 1000;

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
    theta_vec[i] = 0.01;
  }
  model.updateOutputModelInit({ 68, 100, 20 }, theta_vec);

  auto lstm_params = model.getLSTMParams();
  lstm_params.setAllValues(1.0f);
  model.setLSTMParams(lstm_params);

  auto init_params = model.getInitLSTMParams();
  init_params.setAllValues(0.1f);
  model.setInitParams(init_params);

  model.GPUSetup();

  std::vector<float> theta(T::OUTPUT_PARAMS_T::NUM_PARAMS);
  std::fill(theta.begin(), theta.end(), 1);
  model.updateOutputModel({ 18, 3 }, theta);

  Eigen::Matrix<float, T::INPUT_DIM, num_rollouts> inputs;
  inputs = Eigen::Matrix<float, T::INPUT_DIM, num_rollouts>::Ones() * 0.1;
  T::output_array output;

  std::array<float, 5> true_vals = { 10.945133, 11.680509, 11.783686, 11.797728, 11.79963 };

  std::vector<std::array<float, T::INPUT_DIM>> input_arr(num_rollouts);
  std::vector<std::array<float, T::OUTPUT_DIM>> output_arr(num_rollouts);

  T::init_buffer buffer = T::init_buffer::Ones() * 0.1;

  for (int state_index = 0; state_index < num_rollouts; state_index++)
  {
    for (int dim = 0; dim < input_arr[0].size(); dim++)
    {
      input_arr[state_index][dim] = inputs.col(state_index)(dim);
    }
  }

  for (int y_dim = 1; y_dim < 16; y_dim++)
  {
    for (int step = 1; step < 6; step++)
    {
      model.initializeLSTM(buffer);

      for (int i = 0; i < T::HIDDEN_DIM; i++)
      {
        EXPECT_FLOAT_EQ(model.getInitModel()->getHiddenState()[i], 0.99790788);
        EXPECT_FLOAT_EQ(model.getInitModel()->getCellState()[i], 9.2190571);
        EXPECT_FLOAT_EQ(model.getLSTMModel()->getHiddenState()[i], 0.558857381);
        EXPECT_FLOAT_EQ(model.getLSTMModel()->getCellState()[i], 0.558857381);
        EXPECT_FLOAT_EQ(model.getLSTMParams().initial_hidden[i], 0.558857381);
        EXPECT_FLOAT_EQ(model.getLSTMParams().initial_cell[i], 0.558857381);
      }
      launchForwardTestKernel<LSTM, 32>(*model.getLSTMModel(), input_arr, output_arr, y_dim, step);
      for (int point = 0; point < num_rollouts; point++)
      {
        // model.resetInitHiddenCPU();
        model.resetLSTMHiddenCellCPU();
        T::input_array input = inputs.col(point);
        T::output_array output;

        for (int cpu_step = 0; cpu_step < step; cpu_step++)
        {
          model.forward(input, output);
        }
        for (int dim = 0; dim < T::INPUT_DIM; dim++)
        {
          EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        }
        for (int dim = 0; dim < T::OUTPUT_DIM; dim++)
        {
          EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4)
              << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
          EXPECT_TRUE(isfinite(output_arr[point][dim]));
          EXPECT_FLOAT_EQ(output(dim), true_vals[step - 1]) << "at dim " << dim << " step " << step;
        }
      }
    }
  }
}

TEST_F(LSTMLSTMHelperTest, forwardGPUCompareShared)
{
  const int num_rollouts = 1000;

  T model;
  model.GPUSetup();

  std::vector<float> theta_vec(LSTM::OUTPUT_PARAMS_T::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = distribution(generator);
  }
  model.updateOutputModel({ 18, 3 }, theta_vec);

  theta_vec.resize(T::INIT_OUTPUT_PARAMS_T::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = distribution(generator);
  }
  model.updateOutputModelInit({ 68, 100, 20 }, theta_vec);

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

  auto init_params = model.getInitLSTMParams();
  for (int i = 0; i < 60 * 60; i++)
  {
    params.W_im[i] = distribution(generator);
    params.W_fm[i] = distribution(generator);
    params.W_om[i] = distribution(generator);
    params.W_cm[i] = distribution(generator);
  }
  for (int i = 0; i < 8 * 60; i++)
  {
    params.W_ii[i] = distribution(generator);
    params.W_fi[i] = distribution(generator);
    params.W_oi[i] = distribution(generator);
    params.W_ci[i] = distribution(generator);
  }
  for (int i = 0; i < 60; i++)
  {
    params.b_i[i] = distribution(generator);
    params.b_f[i] = distribution(generator);
    params.b_o[i] = distribution(generator);
    params.b_c[i] = distribution(generator);
  }
  model.setInitParams(init_params);

  Eigen::Matrix<float, LSTM::INPUT_DIM, num_rollouts> inputs;
  inputs = Eigen::Matrix<float, LSTM::INPUT_DIM, num_rollouts>::Random();
  LSTM::output_array output;

  std::vector<std::array<float, LSTM::INPUT_DIM>> input_arr(num_rollouts);
  std::vector<std::array<float, 3>> output_arr(num_rollouts);


  for (int state_index = 0; state_index < num_rollouts; state_index++)
  {
    for (int dim = 0; dim < input_arr[0].size(); dim++)
    {
      input_arr[state_index][dim] = inputs.col(state_index)(dim);
    }
  }

  for (int y_dim = 1; y_dim < 2; y_dim++)
  {
    for (int step = 1; step < 6; step++)
    {
      T::init_buffer buffer = T::init_buffer::Random();
      model.initializeLSTM(buffer);

      launchForwardTestKernel<LSTM, 32>(*model.getLSTMModel(), input_arr, output_arr, y_dim, step);
      for (int point = 0; point < num_rollouts; point++)
      {
        // model.resetInitHiddenCPU();
        model.resetLSTMHiddenCellCPU();
        T::input_array input = inputs.col(point);
        T::output_array output;

        for (int cpu_step = 0; cpu_step < step; cpu_step++)
        {
          model.forward(input, output);
        }
        for (int dim = 0; dim < T::INPUT_DIM; dim++)
        {
          EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        }
        for (int dim = 0; dim < T::OUTPUT_DIM; dim++)
        {
          EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4)
              << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
          EXPECT_TRUE(isfinite(output_arr[point][dim]));
        }
      }
    }
  }
}

typedef LSTMHelper<LSTMParams<8, 10>, FNN_PARAMS, false> LSTM2;
typedef LSTMHelper<LSTMParams<8, 60>, FNN_INIT_PARAMS, false> INIT_LSTM2;
typedef LSTMLSTMHelper<INIT_LSTM2, LSTM2, 10> T2;
TEST_F(LSTMLSTMHelperTest, forwardGPUCompareNoShared)
{
  const int num_rollouts = 1000;

  T2 model;
  model.GPUSetup();

  std::vector<float> theta_vec(LSTM2::OUTPUT_PARAMS_T::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = distribution(generator);
  }
  model.updateOutputModel({ 18, 3 }, theta_vec);

  theta_vec.resize(T2::INIT_OUTPUT_PARAMS_T::NUM_PARAMS);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = distribution(generator);
  }
  model.updateOutputModelInit({ 68, 100, 20 }, theta_vec);

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

  auto init_params = model.getInitLSTMParams();
  for (int i = 0; i < 60 * 60; i++)
  {
    params.W_im[i] = distribution(generator);
    params.W_fm[i] = distribution(generator);
    params.W_om[i] = distribution(generator);
    params.W_cm[i] = distribution(generator);
  }
  for (int i = 0; i < 8 * 60; i++)
  {
    params.W_ii[i] = distribution(generator);
    params.W_fi[i] = distribution(generator);
    params.W_oi[i] = distribution(generator);
    params.W_ci[i] = distribution(generator);
  }
  for (int i = 0; i < 60; i++)
  {
    params.b_i[i] = distribution(generator);
    params.b_f[i] = distribution(generator);
    params.b_o[i] = distribution(generator);
    params.b_c[i] = distribution(generator);
  }
  model.setInitParams(init_params);

  Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts> inputs;
  inputs = Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts>::Random();
  LSTM2::output_array output;

  std::vector<std::array<float, LSTM2::INPUT_DIM>> input_arr(num_rollouts);
  std::vector<std::array<float, 3>> output_arr(num_rollouts);


  for (int state_index = 0; state_index < num_rollouts; state_index++)
  {
    for (int dim = 0; dim < input_arr[0].size(); dim++)
    {
      input_arr[state_index][dim] = inputs.col(state_index)(dim);
    }
  }

  for (int y_dim = 1; y_dim < 2; y_dim++)
  {
    for (int step = 1; step < 6; step++)
    {
      T::init_buffer buffer = T2::init_buffer::Random();
      model.initializeLSTM(buffer);

      launchForwardTestKernel<LSTM2, 32>(*model.getLSTMModel(), input_arr, output_arr, y_dim, step);
      for (int point = 0; point < num_rollouts; point++)
      {
        // model.resetInitHiddenCPU();
        model.resetLSTMHiddenCellCPU();
        T2::input_array input = inputs.col(point);
        T2::output_array output;

        for (int cpu_step = 0; cpu_step < step; cpu_step++)
        {
          model.forward(input, output);
        }
        for (int dim = 0; dim < T2::INPUT_DIM; dim++)
        {
          EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        }
        for (int dim = 0; dim < T2::OUTPUT_DIM; dim++)
        {
          EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4)
              << "at index " << point << " with y_dim " << y_dim << " dim " << dim;
          EXPECT_TRUE(isfinite(output_arr[point][dim]));
        }
      }
    }
  }
}

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
