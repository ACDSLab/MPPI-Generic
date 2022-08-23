#include <gtest/gtest.h>

#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>
#include <math.h>

#include <mppi/utils/nn_helpers/lstm_helper.cuh>
// Auto-generated header file
#include <autorally_test_network.h>
#include <mppi/utils/network_helper_kernel_test.cuh>
#include <unsupported/Eigen/NumericalDiff>
#include "mppi/ddp/ddp_dynamics.h"

class LSTMHelperTest : public testing::Test
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

TEST_F(LSTMHelperTest, ParamsConstructor1)
{
  const int hidden_dim = LSTMParams<5, 25>::HIDDEN_DIM;
  const int shared_mem_grd = LSTMParams<5, 25>::SHARED_MEM_REQUEST_GRD;
  const int shared_mem_blk = LSTMParams<5, 25>::SHARED_MEM_REQUEST_BLK;
  const int input_dim = LSTMParams<5, 25>::INPUT_DIM;
  const int hidden_hidden_dim = LSTMParams<5, 25>::HIDDEN_HIDDEN_SIZE;
  const int input_hidden_dim = LSTMParams<5, 25>::INPUT_HIDDEN_SIZE;
  EXPECT_EQ(shared_mem_grd, sizeof(LSTMParams<5, 25>));
  EXPECT_EQ(shared_mem_blk, 25 * 8 + 5);
  EXPECT_EQ(input_dim, 5);
  EXPECT_EQ(hidden_hidden_dim, 25 * 25);
  EXPECT_EQ(input_hidden_dim, 5 * 25);

  LSTMParams<5, 25> params;

  auto W_im = params.W_im;
  auto W_fm = params.W_fm;
  auto W_om = params.W_om;
  auto W_cm = params.W_cm;
  for (int i = 0; i < 25 * 25; i++)
  {
    EXPECT_FLOAT_EQ(W_im[i], 0.0f);
    EXPECT_FLOAT_EQ(W_fm[i], 0.0f);
    EXPECT_FLOAT_EQ(W_cm[i], 0.0f);
    EXPECT_FLOAT_EQ(W_om[i], 0.0f);
  }

  auto W_ii = params.W_ii;
  auto W_fi = params.W_fi;
  auto W_oi = params.W_oi;
  auto W_ci = params.W_ci;
  for (int i = 0; i < 5 * 25; i++)
  {
    EXPECT_FLOAT_EQ(W_ii[i], 0.0f);
    EXPECT_FLOAT_EQ(W_fi[i], 0.0f);
    EXPECT_FLOAT_EQ(W_oi[i], 0.0f);
    EXPECT_FLOAT_EQ(W_ci[i], 0.0f);
  }

  auto b_i = params.b_i;
  auto b_f = params.b_f;
  auto b_o = params.b_o;
  auto b_c = params.b_c;
  for (int i = 0; i < 25; i++)
  {
    EXPECT_FLOAT_EQ(b_i[i], 0.0f);
    EXPECT_FLOAT_EQ(b_f[i], 0.0f);
    EXPECT_FLOAT_EQ(b_o[i], 0.0f);
    EXPECT_FLOAT_EQ(b_c[i], 0.0f);
  }

  auto init_hidden = params.initial_hidden;
  auto init_cell = params.initial_cell;
  for (int i = 0; i < 25; i++)
  {
    EXPECT_FLOAT_EQ(init_hidden[i], 0.0f);
    EXPECT_FLOAT_EQ(init_cell[i], 0.0f);
  }
}

// TEST_F(LSTMHelperTest, ParamsConstructor2)
// {
//   int total_amount = 0;
//   total_amount += LSTMHelper<LSTMParams<8, 20>, FNNHelper<FNNParams<20, 3>>>::SHARED_MEM_REQUEST_GRD;
//   total_amount += LSTMHelper<LSTMParams<8, 20>, FNNHelper<FNNParams<20, 3>>>::SHARED_MEM_REQUEST_BLK * 8;
//   total_amount += LSTMHelper<LSTMParams<3, 10>, FNNHelper<FNNParams<10, 1>>>::SHARED_MEM_REQUEST_GRD;
//   total_amount += LSTMHelper<LSTMParams<3, 10>, FNNHelper<FNNParams<10, 1>>>::SHARED_MEM_REQUEST_BLK * 8;
//
//   EXPECT_LT(total_amount * 4, 49152);
// }

TEST_F(LSTMHelperTest, BindStream)
{
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  LSTMHelper<LSTMParams<5, 25>, FNNHelper<FNNParams<25, 30, 3>>> helper(stream);

  EXPECT_EQ(helper.stream_, stream);
}

using LSTM = LSTMHelper<LSTMParams<8, 20>, FNNHelper<FNNParams<28, 3>>>;
TEST_F(LSTMHelperTest, GPUSetupAndParamsCheck)
{
  LSTM model;

  std::vector<float> theta_vec(87);
  for (int i = 0; i < 87; i++)
  {
    theta_vec[i] = distribution(generator);
  }
  model.output_nn_->updateModel({ 28, 3 }, theta_vec);

  int grid_dim = 5;

  std::vector<LSTM::LSTM_PARAMS_T> lstm_params(grid_dim);
  std::vector<LSTM::LSTM_PARAMS_T> shared_lstm_params(grid_dim);
  std::vector<LSTM::OUTPUT_FNN_T::NN_PARAMS_T> fnn_params(grid_dim);
  std::vector<LSTM::OUTPUT_FNN_T::NN_PARAMS_T> shared_fnn_params(grid_dim);

  EXPECT_EQ(model.GPUMemStatus_, false);
  EXPECT_EQ(model.network_d_, nullptr);
  EXPECT_NE(model.output_nn_, nullptr);

  model.GPUSetup();

  EXPECT_EQ(model.GPUMemStatus_, true);
  EXPECT_NE(model.network_d_, nullptr);

  // launch kernel
  launchParameterCheckTestKernel<LSTM>(model, lstm_params, shared_lstm_params, fnn_params, shared_fnn_params);

  for (int grid = 0; grid < grid_dim; grid++)
  {
    // ensure that the output nn matches
    for (int i = 0; i < 87; i++)
    {
      EXPECT_FLOAT_EQ(fnn_params[grid].theta[i], theta_vec[i]) << "at grid " << grid << " at index " << i;
      EXPECT_FLOAT_EQ(shared_fnn_params[grid].theta[i], theta_vec[i]) << "at grid " << grid << "at index " << i;
    }
    EXPECT_EQ(fnn_params[grid].stride_idcs[0], 0);
    EXPECT_EQ(fnn_params[grid].stride_idcs[1], 84);
    EXPECT_EQ(shared_fnn_params[grid].stride_idcs[0], 0) << "at grid " << grid;
    EXPECT_EQ(shared_fnn_params[grid].stride_idcs[1], 84) << "at grid " << grid;

    EXPECT_EQ(fnn_params[grid].net_structure[0], 28);
    EXPECT_EQ(fnn_params[grid].net_structure[1], 3);
    EXPECT_EQ(shared_fnn_params[grid].net_structure[0], 28) << "at grid " << grid;
    ;
    EXPECT_EQ(shared_fnn_params[grid].net_structure[1], 3) << "at grid " << grid;
    ;

    for (int i = 0; i < 20 * 20; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].W_im[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_fm[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_cm[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_om[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_im[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_fm[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_cm[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_om[i], 0.0f);
    }

    for (int i = 0; i < 8 * 20; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].W_ii[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_fi[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_oi[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].W_ci[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_ii[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_fi[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_oi[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].W_ci[i], 0.0f);
    }

    for (int i = 0; i < 20; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].b_i[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].b_f[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].b_o[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].b_c[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_i[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_f[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_o[i], 0.0f);
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].b_c[i], 0.0f);
      EXPECT_FLOAT_EQ(lstm_params[grid].initial_hidden[i], 0.0f) << "at index " << i;
      EXPECT_FLOAT_EQ(lstm_params[grid].initial_cell[i], 0.0f) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].initial_hidden[i], 0.0f) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].initial_cell[i], 0.0f) << "at index " << i;
    }
  }
}

TEST_F(LSTMHelperTest, UpdateModel)
{
  LSTM model;

  int grid_dim = 5;

  std::vector<float> theta_vec(87);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = distribution(generator);
  }
  model.updateOutputModel({ 28, 3 }, theta_vec);

  auto params = model.getLSTMParams();
  for (int i = 0; i < 20 * 20; i++)
  {
    params.W_im[i] = distribution(generator);
    params.W_fm[i] = distribution(generator);
    params.W_om[i] = distribution(generator);
    params.W_cm[i] = distribution(generator);
  }
  for (int i = 0; i < 8 * 20; i++)
  {
    params.W_ii[i] = distribution(generator);
    params.W_fi[i] = distribution(generator);
    params.W_oi[i] = distribution(generator);
    params.W_ci[i] = distribution(generator);
  }
  for (int i = 0; i < 20; i++)
  {
    params.b_i[i] = distribution(generator);
    params.b_f[i] = distribution(generator);
    params.b_o[i] = distribution(generator);
    params.b_c[i] = distribution(generator);
    params.initial_hidden[i] = distribution(generator);
    params.initial_cell[i] = distribution(generator);
  }
  model.updateLSTM(params);

  for (int i = 0; i < 20; i++)
  {
    EXPECT_FLOAT_EQ(model.getHiddenState()[i], params.initial_hidden[i]);
    EXPECT_FLOAT_EQ(model.getCellState()[i], params.initial_cell[i]);
  }

  std::vector<LSTM::LSTM_PARAMS_T> lstm_params(grid_dim);
  std::vector<LSTM::LSTM_PARAMS_T> shared_lstm_params(grid_dim);
  std::vector<LSTM::OUTPUT_FNN_T::NN_PARAMS_T> fnn_params(grid_dim);
  std::vector<LSTM::OUTPUT_FNN_T::NN_PARAMS_T> shared_fnn_params(grid_dim);

  EXPECT_EQ(model.GPUMemStatus_, false);
  EXPECT_EQ(model.network_d_, nullptr);
  EXPECT_NE(model.output_nn_, nullptr);

  model.GPUSetup();

  EXPECT_EQ(model.GPUMemStatus_, true);
  EXPECT_NE(model.network_d_, nullptr);

  // launch kernel
  launchParameterCheckTestKernel<LSTM>(model, lstm_params, shared_lstm_params, fnn_params, shared_fnn_params);

  for (int grid = 0; grid < grid_dim; grid++)
  {
    // ensure that the output nn matches
    for (int i = 0; i < 87; i++)
    {
      EXPECT_FLOAT_EQ(fnn_params[grid].theta[i], theta_vec[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_fnn_params[grid].theta[i], theta_vec[i]) << "at index " << i;
    }
    EXPECT_EQ(fnn_params[grid].stride_idcs[0], 0);
    EXPECT_EQ(fnn_params[grid].stride_idcs[1], 84);
    EXPECT_EQ(shared_fnn_params[grid].stride_idcs[0], 0) << "at grid " << grid;
    EXPECT_EQ(shared_fnn_params[grid].stride_idcs[1], 84) << "at grid " << grid;

    EXPECT_EQ(fnn_params[grid].net_structure[0], 28);
    EXPECT_EQ(fnn_params[grid].net_structure[1], 3);
    EXPECT_EQ(shared_fnn_params[grid].net_structure[0], 28) << "at grid " << grid;
    ;
    EXPECT_EQ(shared_fnn_params[grid].net_structure[1], 3) << "at grid " << grid;
    ;

    for (int i = 0; i < 20 * 20; i++)
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

    for (int i = 0; i < 8 * 20; i++)
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

    for (int i = 0; i < 20; i++)
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

    for (int i = 0; i < 20; i++)
    {
      EXPECT_FLOAT_EQ(lstm_params[grid].initial_hidden[i], params.initial_hidden[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(lstm_params[grid].initial_cell[i], params.initial_cell[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].initial_hidden[i], params.initial_hidden[i]) << "at index " << i;
      EXPECT_FLOAT_EQ(shared_lstm_params[grid].initial_cell[i], params.initial_cell[i]) << "at index " << i;
    }
  }
}

// TEST_F(LSTMHelperTest, LoadModelNPZTest)
// {
//   LSTMHelper<LSTMParams<6, 32, 32, 4>> model;
//   model.GPUSetup();
//
//   std::string path = mppi::tests::test_load_nn_file;
//   if (!fileExists(path))
//   {
//     std::cerr << "Could not load neural net model at path: " << path.c_str();
//     exit(-1);
//   }
//   cnpy::npz_t param_dict = cnpy::npz_load(path);
//   model.loadParams(param_dict);
//
//   // check CPU
//   for (int i = 0; i < 1412; i++)
//   {
//     EXPECT_FLOAT_EQ(model.getTheta()[i], i) << "failed at index " << i;
//   }
//
//   std::array<float, 1412> theta_result = {};
//   std::array<int, 6> stride_result = {};
//   std::array<int, 4> net_structure_result = {};
//   std::array<float, 1412> shared_theta_result = {};
//   std::array<int, 6> shared_stride_result = {};
//   std::array<int, 4> shared_net_structure_result = {};
//
//   // launch kernel
//   launchParameterCheckTestKernel<LSTMHelper<LSTMParams<6, 32, 32, 4>>, 1412, 6, 4>(
//       model, theta_result, stride_result, net_structure_result, shared_theta_result, shared_stride_result,
//       shared_net_structure_result);
//
//   for (int i = 0; i < 1412; i++)
//   {
//     EXPECT_FLOAT_EQ(shared_theta_result[i], i) << "failed at index " << i;
//   }
// }

TEST_F(LSTMHelperTest, forwardCPU)
{
  LSTM model;

  std::vector<float> theta_vec(87);
  for (int i = 0; i < theta_vec.size(); i++)
  {
    theta_vec[i] = 1.0;
  }
  model.updateOutputModel({ 28, 3 }, theta_vec);

  auto params = model.getLSTMParams();
  params.setAllValues(1.0f);
  model.updateLSTM(params);

  LSTM::input_array input = LSTM::input_array::Ones();
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
  EXPECT_FLOAT_EQ(output[0], 28.999756);
  EXPECT_FLOAT_EQ(output[1], 28.999756);
  EXPECT_FLOAT_EQ(output[2], 28.999756);
}

using LSTM2 = LSTMHelper<LSTMParams<8, 20, 0>, FNNHelper<FNNParams<28, 3>>>;
TEST_F(LSTMHelperTest, forwardGPU)
{
  const int num_rollouts = 1;

  LSTM2 model;
  model.GPUSetup();

  std::vector<float> theta(87);
  std::fill(theta.begin(), theta.end(), 1);
  model.updateOutputModel({ 28, 3 }, theta);

  auto params = model.getLSTMParams();
  for (int i = 0; i < 20 * 20; i++)
  {
    params.W_im[i] = 1.0;
    params.W_fm[i] = 1.0;
    params.W_om[i] = 1.0;
    params.W_cm[i] = 1.0;
  }
  for (int i = 0; i < 8 * 20; i++)
  {
    params.W_ii[i] = 1.0;
    params.W_fi[i] = 1.0;
    params.W_oi[i] = 1.0;
    params.W_ci[i] = 1.0;
  }
  for (int i = 0; i < 20; i++)
  {
    params.b_i[i] = 1.0;
    params.b_f[i] = 1.0;
    params.b_o[i] = 1.0;
    params.b_c[i] = 1.0;
    params.initial_hidden[i] = 1.0;
    params.initial_cell[i] = 1.0;
  }
  model.updateLSTM(params);

  Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts> inputs;
  inputs = Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts>::Ones();
  LSTM2::output_array output;

  std::array<float, 5> true_vals = { 28.28055, 28.901096, 28.986588, 28.998184, 28.999756 };

  std::vector<std::array<float, LSTM2::INPUT_DIM>> input_arr(num_rollouts);
  std::vector<std::array<float, 3>> output_arr(num_rollouts);

  for (int y_dim = 1; y_dim < 16; y_dim++)
  {
    for (int state_index = 0; state_index < num_rollouts; state_index++)
    {
      for (int dim = 0; dim < input_arr[0].size(); dim++)
      {
        input_arr[state_index][dim] = inputs.col(state_index)(dim);
      }
    }

    for (int step = 1; step < 6; step++)
    {
      launchForwardTestKernel<LSTM2, 32>(model, input_arr, output_arr, y_dim, step);
      for (int point = 0; point < num_rollouts; point++)
      {
        model.resetHiddenCellCPU();
        LSTM2::input_array input = inputs.col(point);
        LSTM2::output_array output;

        for (int cpu_step = 0; cpu_step < step; cpu_step++)
        {
          model.forward(input, output);
        }
        for (int dim = 0; dim < LSTM2::INPUT_DIM; dim++)
        {
          EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        }
        for (int dim = 0; dim < LSTM2::OUTPUT_DIM; dim++)
        {
          EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
          EXPECT_TRUE(isfinite(output_arr[point][dim]));
          EXPECT_FLOAT_EQ(output(dim), true_vals[step - 1]) << "at dim " << dim << " step " << step;
        }
      }
    }
  }
}

TEST_F(LSTMHelperTest, forwardGPUCompare)
{
  const int num_rollouts = 1000;

  LSTM2 model;
  model.GPUSetup();

  std::vector<float> theta(87);
  std::fill(theta.begin(), theta.end(), 1);
  model.updateOutputModel({ 28, 3 }, theta);

  auto params = model.getLSTMParams();
  for (int i = 0; i < 20 * 20; i++)
  {
    params.W_im[i] = 1.0;
    params.W_fm[i] = 1.0;
    params.W_om[i] = 1.0;
    params.W_cm[i] = 1.0;
  }
  for (int i = 0; i < 8 * 20; i++)
  {
    params.W_ii[i] = 1.0;
    params.W_fi[i] = 1.0;
    params.W_oi[i] = 1.0;
    params.W_ci[i] = 1.0;
  }
  for (int i = 0; i < 20; i++)
  {
    params.b_i[i] = 1.0;
    params.b_f[i] = 1.0;
    params.b_o[i] = 1.0;
    params.b_c[i] = 1.0;
    params.initial_hidden[i] = 1.0;
    params.initial_cell[i] = 1.0;
  }
  model.updateLSTM(params);

  Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts> inputs;
  inputs = Eigen::Matrix<float, LSTM2::INPUT_DIM, num_rollouts>::Random();
  LSTM2::output_array output;

  std::vector<std::array<float, LSTM2::INPUT_DIM>> input_arr(num_rollouts);
  std::vector<std::array<float, 3>> output_arr(num_rollouts);

  for (int y_dim = 1; y_dim < 16; y_dim++)
  {
    for (int state_index = 0; state_index < num_rollouts; state_index++)
    {
      for (int dim = 0; dim < input_arr[0].size(); dim++)
      {
        input_arr[state_index][dim] = inputs.col(state_index)(dim);
      }
    }
    for (int step = 1; step < 6; step++)
    {
      launchForwardTestKernel<LSTM2, 32>(model, input_arr, output_arr, y_dim, step);
      for (int point = 0; point < num_rollouts; point++)
      {
        model.resetHiddenCellCPU();
        LSTM2::input_array input = inputs.col(point);
        LSTM2::output_array output;

        for (int cpu_step = 0; cpu_step < step; cpu_step++)
        {
          model.forward(input, output);
        }
        for (int dim = 0; dim < LSTM2::INPUT_DIM; dim++)
        {
          EXPECT_NEAR(input(dim), input_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        }
        for (int dim = 0; dim < LSTM2::OUTPUT_DIM; dim++)
        {
          EXPECT_NEAR(output(dim), output_arr[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
          EXPECT_TRUE(isfinite(output_arr[point][dim]));
        }
      }
    }
  }
}

// TEST_F(LSTMHelperTest, TestComputeGradComputationFinite)
// {
//   LSTMHelper<LSTMParams<6, 32, 32, 4>> model;
//   std::vector<float> theta(1412);
//   for (int i = 0; i < 1412; i++)
//   {
//     theta[i] = distribution(generator);
//   }
//   model.updateModel({ 6, 32, 32, 4 }, theta);
//
//   LSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx numeric_jac;
//   LSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx analytic_jac;
//
//   for (int i = 0; i < 1000; i++)
//   {
//     LSTMHelper<LSTMParams<6, 32, 32, 4>>::input_array input;
//     input = LSTMHelper<LSTMParams<6, 32, 32, 4>>::input_array::Random();
//
//     model.computeGrad(input, analytic_jac);
//     EXPECT_TRUE(analytic_jac.allFinite());
//   }
// }
//
// TEST_F(LSTMHelperTest, TestComputeGradComputationCompare)
// {
//   GTEST_SKIP();
//   LSTMHelper<LSTMParams<6, 32, 32, 4>> model;
//   std::vector<float> theta(1412);
//   for (int i = 0; i < 1412; i++)
//   {
//     theta[i] = distribution(generator);
//   }
//   model.updateModel({ 6, 32, 32, 4 }, theta);
//
//   LSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx numeric_jac;
//   LSTMHelper<LSTMParams<6, 32, 32, 4>>::dfdx analytic_jac;
//
//   LSTMHelper<LSTMParams<6, 32, 32, 4>>::input_array input;
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
