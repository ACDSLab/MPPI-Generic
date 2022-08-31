//
// Created by bvlahov3 on 6/15/21.
//

// #include <gtest/gtest.h>
// #include <mppi/dynamics/LSTM/LSTM_model.cuh>
// #include <mppi/dynamics/autorally/ar_nn_model.cuh>
// #include <mppi/dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>
// #include <stdio.h>
// #include <math.h>
// #include <mppi/controllers/MPPI/mppi_controller.cuh>
// #include <mppi/cost_functions/autorally/ar_standard_cost.cuh>
// #include <mppi/feedback_controllers/DDP/ddp.cuh>
//
// // Auto-generated header files
// #include <autorally_test_network.h>
// #include <autorally_test_map.h>
//
// #include <chrono>
//
// const int CONTROL_DIM = 2;
// const int HIDDEN_DIM = 15;
// const int BUFFER = 11;
// const int INIT_DIM = 200;
// // typedef LSTMModel<7, CONTROL_DIM, 3, 32> DYNAMICS;
// using DYNAMICS = LSTMModel<7, CONTROL_DIM, 3, HIDDEN_DIM, BUFFER, INIT_DIM>;
// using DYN_PARAMS = DYNAMICS::DYN_PARAMS_T;
//
// void assert_float_array_eq(float* pred, float* gt, int max)
// {
//   for (int i = 0; i < max; i++)
//   {
//     float pred_temp = pred[i];
//     float gt_temp = gt[i];
//     ASSERT_NEAR(pred[i], gt[i], 0.01 * abs(gt[i]))
//         << "Expected " << gt[i] << " but saw " << pred[i] << " at " << i << std::endl;
//   }
// }
//
// // Struct to look at protected variables of dynamics model
// struct ModelExposer : DYNAMICS
// {
//   DYNAMICS* model_;
//   ModelExposer(DYNAMICS* model)
//   {
//     model_ = model;
//   }
//   __host__ __device__ DYN_PARAMS* getParamsPointer()
//   {
//     return (DYN_PARAMS*)&((ModelExposer*)model_)->params_;
//   }
// };
//
// __global__ void access_params(ModelExposer* model)
// {
//   DYN_PARAMS* params = model->getParamsPointer();
//   printf("Check if copy_everything is true: %d\n", params->copy_everything);
//   printf("SHARED_MEM_REQUEST_BLK: %d\n", params->SHARED_MEM_REQUEST_BLK);
//   printf("W_im: %p\n", params->W_im);
//   printf("W_im[0]: %f\n", params->W_im[0]);
//   printf("W_fm[0]: %f\n", params->W_fm[0]);
// }
//
// template <class DYN_T, int NUM_ROLLOUTS = 1, int BLOCKSIZE_X = 1, int BLOCKSIZE_Z = 1>
// __global__ void run_dynamics(DYN_T* dynamics, float* initial_state, float* control, float* state_der)
// {
//   int thread_idx = threadIdx.x;
//   int thread_idy = threadIdx.y;
//   int thread_idz = threadIdx.z;
//   int block_idx = blockIdx.x;
//   int global_idx = blockDim.x * block_idx + thread_idx;
//
//   // Create shared state and control arrays
//   __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
//   __shared__ float y_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
//   __shared__ float x_next_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
//   __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
//   __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
//   __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
//   __shared__ float sigma_u[DYN_T::CONTROL_DIM];
//   __shared__ int crash_status_shared[BLOCKSIZE_X * BLOCKSIZE_Z];
//
//   // Create a shared array for the dynamics model to use
//   __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK * BLOCKSIZE_X *
//   BLOCKSIZE_Z];
//
//   float* x;
//   float* xdot;
//   float* x_next;
//   float* x_temp;
//   float* y;
//   float* u;
//   float dt = 0.02;
//   // float* du;
//   // int* crash_status;
//   if (global_idx < NUM_ROLLOUTS)
//   {
//     x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
//     x_next = &x_next_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
//     y = &y_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
//     xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
//     u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
//     // du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
//     // crash_status = &crash_status_shared[thread_idz*blockDim.x + thread_idx];
//     // crash_status[0] = 0; // We have not crashed yet as of the first trajectory.
//
//     //__syncthreads();
//     for (int i = threadIdx.y; i < DYN_T::STATE_DIM; i += blockDim.y)
//     {
//       x[i] = initial_state[i];
//     }
//     for (int i = threadIdx.y; i < DYN_T::CONTROL_DIM; i += blockDim.y)
//     {
//       u[i] = control[i];
//     }
//     __syncthreads();
//     /*<----Start of simulation loop-----> */
//     dynamics->initializeDynamics(x, u, y, theta_s, 0.0, 0.0);
//     // dynamics->computeStateDeriv(x, u, xdot, theta_s);
//     dynamics->step(x, x_next, xdot, u, y, theta_s, 0.0, 0.1);
//     // __syncthreads();
//     // dynamics->updateState(x, xdot, dt);
//     // __syncthreads();
//     // u[0] = -2.2619e-03;
//     // u[1] = 2.3031e-01;
//     // dynamics->computeStateDeriv(x, u, xdot, theta_s);
//
//     __syncthreads();
//     for (int i = threadIdx.y; i < DYN_T::STATE_DIM; i += blockDim.y)
//     {
//       state_der[global_idx * DYN_T::STATE_DIM + i] = xdot[i];
//     }
//   }
// }
//
// class LSTMDynamicsTest : public ::testing::Test
// {
// public:
//   cudaStream_t stream;
//
//   std::array<float2, CONTROL_DIM> u_constraint = {};
//
//   virtual void SetUp()
//   {
//     HANDLE_ERROR(cudaStreamCreate(&stream));
//     u_constraint[0].x = -1.0;
//     u_constraint[0].y = 1.0;
//
//     u_constraint[1].x = -2.0;
//     u_constraint[1].y = 2.0;
//   }
// };
//
// TEST_F(LSTMDynamicsTest, BindStreamControlRanges)
// {
//   DYNAMICS model(u_constraint, stream);
//   EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";
//
//   HANDLE_ERROR(cudaStreamDestroy(stream));
// }
//
// TEST_F(LSTMDynamicsTest, updateBuffer)
// {
//   DYNAMICS model(u_constraint, stream);
//   auto params = model.getParams();
//
//   Eigen::Matrix<float, 9, 11> new_buffer;
//   new_buffer.setRandom();
//   params.updateBuffer(new_buffer);
//
//   Eigen::Matrix<float, 6, 11> result_buffer;
//   result_buffer.row(0) = new_buffer.row(3);
//   result_buffer.row(1) = new_buffer.row(4);
//   result_buffer.row(2) = new_buffer.row(5);
//   result_buffer.row(3) = new_buffer.row(6);
//   result_buffer.row(4) = new_buffer.row(7);
//   result_buffer.row(5) = new_buffer.row(8);
//
//   for (int col = 0; col < result_buffer.cols(); col++)
//   {
//     for (int row = 0; row < result_buffer.rows(); row++)
//     {
//       int index = col * result_buffer.rows() + row;
//       EXPECT_FLOAT_EQ(result_buffer.data()[index], params.buffer[index]) << "row " << row << " col " << col;
//     }
//   }
// }
//
// TEST_F(LSTMDynamicsTest, CopyParams)
// {
//   DYNAMICS model(u_constraint, stream);
//
//   model.GPUSetup();
//   auto dyn_params = model.getParams();
//   dyn_params.W_fm[0] = 5.0;
//   dyn_params.copy_everything = true;
//   dyn_params.W_hidden_input.get()[0] = 13;
//   model.setParams(dyn_params);
//   ModelExposer access_cpu_model(&model);
//
//   std::cout << "Params Size: " << sizeof(dyn_params) << " bytes" << std::endl;
//   // std::cout << "Weight Size: " << sizeof(dyn_params.W_im) / sizeof(float) << std::endl;
//   std::cout << "Weight_im: " << dyn_params.W_im << std::endl;
//   std::cout << "Hidden State Initialization Weight references: "
//             << access_cpu_model.getParamsPointer()->W_hidden_input.use_count() << std::endl;
//   std::cout << "Hidden State Initialization Weight[0]: " <<
//   access_cpu_model.getParamsPointer()->W_hidden_input.get()[0]
//             << std::endl;
//
//   dim3 dimBlock(1, 1, 1);
//   dim3 dimGrid(1, 1, 1);
//   ModelExposer access_gpu_model(model.model_d_);
//   ModelExposer* access_gpu_model_d_;
//   HANDLE_ERROR(cudaMalloc((void**)&access_gpu_model_d_, sizeof(ModelExposer)));
//   HANDLE_ERROR(
//       cudaMemcpyAsync(access_gpu_model_d_, &access_gpu_model, sizeof(ModelExposer), cudaMemcpyHostToDevice, stream));
//   access_params<<<dimGrid, dimBlock, 0, stream>>>(access_gpu_model_d_);
//   HANDLE_ERROR(cudaStreamSynchronize(stream));
//
//   EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";
//
//   HANDLE_ERROR(cudaStreamDestroy(stream));
// }
//
// TEST_F(LSTMDynamicsTest, LoadWeights)
// {
//   DYNAMICS model(u_constraint, stream);
//
//   int BUFFER_SIZE = 11 * 6;
//   const int num_rollouts = 10;
//   const int blocksize_x = 32;
//
//   model.GPUSetup();
//   model.loadParams(mppi::tests::autorally_lstm_network_file, mppi::tests::autorally_hidden_network_file,
//                    mppi::tests::autorally_cell_network_file, mppi::tests::autorally_output_network_file);
//
//   std::vector<float> x_0 = { 2.9642e-04, 5.7054e+00, 1.1859e-03, 1.3721e-01, 2.4944e-02, 1.2798e-01 };
//   std::vector<float> x_1 = { 7.8346e-04, 5.6928e+00, -1.4520e-02, 1.7258e-01, -3.1522e-03, 8.4512e-02 };
//   std::vector<float> x_2 = { 7.5389e-04, 5.6884e+00, -1.9062e-02, 4.5813e-04, -2.3523e-02, 6.8172e-02 };
//   std::vector<float> x_3 = { 1.5670e-03, 5.6779e+00, 5.7993e-03, -9.1165e-02, -2.5202e-02, 9.0036e-02 };
//   std::vector<float> x_4 = { 2.0307e-03, 5.6623e+00, 3.5971e-02, -1.4233e-01, -1.4520e-02, 1.3751e-01 };
//   std::vector<float> x_5 = { 6.6427e-04, 5.6565e+00, 4.3000e-02, -2.1955e-02, -2.1740e-03, 1.9203e-01 };
//   std::vector<float> x_6 = { 2.1942e-04, 5.6636e+00, 1.6840e-02, -5.7120e-03, 3.3988e-03, 2.3751e-01 };
//   std::vector<float> x_7 = { 6.9824e-04, 5.6656e+00, -4.1707e-04, -4.3693e-02, 2.9118e-03, 2.6795e-01 };
//   std::vector<float> x_8 = { 1.2957e-03, 5.6861e+00, 2.8441e-03, -1.1037e-01, -6.0677e-04, 2.7991e-01 };
//   std::vector<float> x_9 = { 8.7452e-04, 5.7010e+00, 2.2052e-02, -2.5667e-02, -4.2457e-03, 2.7212e-01 };
//   std::vector<float> x_10 = { 7.2980e-04, 5.7185e+00, 2.0501e-02, -4.2951e-02, -5.5691e-03, 2.5228e-01 };
//
//   Eigen::Matrix<float, 9, 11> new_buffer;
//   // set to be random so there should be an issue if we incorrectly index
//   new_buffer.setRandom();
//
//   new_buffer.block<4, 1>(3, 0) = Eigen::Matrix<float, 4, 1>(x_0.data());
//   new_buffer.block<2, 1>(7, 0) = Eigen::Matrix<float, 2, 1>(x_0.data() + 4);
//
//   new_buffer.block<4, 1>(3, 1) = Eigen::Matrix<float, 4, 1>(x_1.data());
//   new_buffer.block<2, 1>(7, 1) = Eigen::Matrix<float, 2, 1>(x_1.data() + 4);
//
//   new_buffer.block<4, 1>(3, 2) = Eigen::Matrix<float, 4, 1>(x_2.data());
//   new_buffer.block<2, 1>(7, 2) = Eigen::Matrix<float, 2, 1>(x_2.data() + 4);
//
//   new_buffer.block<4, 1>(3, 3) = Eigen::Matrix<float, 4, 1>(x_3.data());
//   new_buffer.block<2, 1>(7, 3) = Eigen::Matrix<float, 2, 1>(x_3.data() + 4);
//
//   new_buffer.block<4, 1>(3, 4) = Eigen::Matrix<float, 4, 1>(x_4.data());
//   new_buffer.block<2, 1>(7, 4) = Eigen::Matrix<float, 2, 1>(x_4.data() + 4);
//
//   new_buffer.block<4, 1>(3, 5) = Eigen::Matrix<float, 4, 1>(x_5.data());
//   new_buffer.block<2, 1>(7, 5) = Eigen::Matrix<float, 2, 1>(x_5.data() + 4);
//
//   new_buffer.block<4, 1>(3, 6) = Eigen::Matrix<float, 4, 1>(x_6.data());
//   new_buffer.block<2, 1>(7, 6) = Eigen::Matrix<float, 2, 1>(x_6.data() + 4);
//
//   new_buffer.block<4, 1>(3, 7) = Eigen::Matrix<float, 4, 1>(x_7.data());
//   new_buffer.block<2, 1>(7, 7) = Eigen::Matrix<float, 2, 1>(x_7.data() + 4);
//
//   new_buffer.block<4, 1>(3, 8) = Eigen::Matrix<float, 4, 1>(x_8.data());
//   new_buffer.block<2, 1>(7, 8) = Eigen::Matrix<float, 2, 1>(x_8.data() + 4);
//
//   new_buffer.block<4, 1>(3, 9) = Eigen::Matrix<float, 4, 1>(x_9.data());
//   new_buffer.block<2, 1>(7, 9) = Eigen::Matrix<float, 2, 1>(x_9.data() + 4);
//
//   new_buffer.block<4, 1>(3, 10) = Eigen::Matrix<float, 4, 1>(x_10.data());
//   new_buffer.block<2, 1>(7, 10) = Eigen::Matrix<float, 2, 1>(x_10.data() + 4);
//
//   auto params = model.getParams();
//   params.updateBuffer(new_buffer);
//   model.setParams(params);
//
//   model.paramsToDevice();
//   DYN_PARAMS dyn_params = model.getParams();
//
//   std::cout << "BUFFER State after update:\n";
//   int col = 0;
//   float* buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_0[row]) << "row " << row << " col " << col;
//   }
//   col = 1;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_1[row]) << "row " << row << " col " << col;
//   }
//   col = 2;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_2[row]) << "row " << row << " col " << col;
//   }
//   col = 3;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_3[row]) << "row " << row << " col " << col;
//   }
//   col = 4;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_4[row]) << "row " << row << " col " << col;
//   }
//   col = 5;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_5[row]) << "row " << row << " col " << col;
//   }
//   col = 6;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_6[row]) << "row " << row << " col " << col;
//   }
//   col = 7;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_7[row]) << "row " << row << " col " << col;
//   }
//   col = 8;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_8[row]) << "row " << row << " col " << col;
//   }
//   col = 9;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_9[row]) << "row " << row << " col " << col;
//   }
//   col = 10;
//   buffer_index = dyn_params.buffer + col * 6;
//   for (int row = 0; row < 6; row++)
//   {
//     EXPECT_FLOAT_EQ(buffer_index[row], x_10[row]) << "row " << row << " col " << col;
//   }
//
//   std::cout << "Initial Hidden State:\n";
//   for (int i = 0; i < HIDDEN_DIM; i++)
//   {
//     std::cout << dyn_params.initial_hidden[i] << ", ";
//   }
//   std::cout << std::endl;
//   float python_initial_hidden[HIDDEN_DIM] = { 1.5683, -1.9951, 1.1584,  -0.7979, 1.4718,  0.4733, -0.0247, -2.4777,
//                                               0.7693, 0.1656,  -1.2776, -0.7229, -0.0814, 0.2607, 0.2293 };
//   float python_initial_cell[HIDDEN_DIM] = { -1.3588, -0.3638, 0.4151, 0.3086, -0.4807, 0.1482,  0.2792, 0.2688,
//                                             -0.5454, -1.4281, 0.2832, 0.3822, -0.1447, -0.4664, 1.6805 };
//   assert_float_array_eq(dyn_params.initial_hidden, python_initial_hidden, HIDDEN_DIM);
//   assert_float_array_eq(dyn_params.initial_cell, python_initial_cell, HIDDEN_DIM);
//
//   float initial_state_cpu[DYNAMICS::STATE_DIM] = { 0.0, 0.0, 0.0, 7.2980e-04, 5.7185e+00, 2.0501e-02, -4.2951e-02 };
//   float control_cpu[DYNAMICS::CONTROL_DIM] = { -5.5691e-03, 2.5228e-01 };
//   float state_der_cpu[num_rollouts * DYNAMICS::STATE_DIM] = { 0.0 };
//
//   float* initial_state_gpu;
//   float* control_gpu;
//   float* state_der_gpu;
//   HANDLE_ERROR(cudaMalloc((void**)&initial_state_gpu, sizeof(float) * DYNAMICS::STATE_DIM));
//   HANDLE_ERROR(cudaMalloc((void**)&control_gpu, sizeof(float) * DYNAMICS::CONTROL_DIM));
//   HANDLE_ERROR(cudaMalloc((void**)&state_der_gpu, sizeof(float) * num_rollouts * DYNAMICS::STATE_DIM));
//   HANDLE_ERROR(cudaMemcpyAsync(initial_state_gpu, initial_state_cpu, sizeof(float) * DYNAMICS::STATE_DIM,
//                                cudaMemcpyHostToDevice, stream));
//   HANDLE_ERROR(
//       cudaMemcpyAsync(control_gpu, control_cpu, sizeof(float) * DYNAMICS::CONTROL_DIM, cudaMemcpyHostToDevice,
//       stream));
//   const int gridsize_x = (num_rollouts - 1) / blocksize_x + 1;
//   dim3 dimBlock(blocksize_x, 16, 1);
//   dim3 dimGrid(gridsize_x, 1, 1);
//   std::cout << "Launching dynamics kernel" << std::endl;
//   run_dynamics<DYNAMICS, num_rollouts, blocksize_x>
//       <<<dimGrid, dimBlock, 0, stream>>>(model.model_d_, initial_state_gpu, control_gpu, state_der_gpu);
//   HANDLE_ERROR(cudaMemcpyAsync(state_der_cpu, state_der_gpu, sizeof(float) * num_rollouts * DYNAMICS::STATE_DIM,
//                                cudaMemcpyDeviceToHost, stream));
//   HANDLE_ERROR(cudaStreamSynchronize(stream));
//   std::cout << "Finished dynamics kernel" << std::endl;
//   std::cout << "State Der:\n";
//   for (int i = 0; i < num_rollouts; i++)
//   {
//     std::cout << "\tRollout " << i << ": ";
//     for (int j = 0; j < DYNAMICS::STATE_DIM; j++)
//     {
//       std::cout << state_der_cpu[i * DYNAMICS::STATE_DIM + j] << ", ";
//     }
//     std::cout << std::endl;
//   }
//   float expected_state_deriv[4] = { -0.2720, 0.8784, -1.1101, 1.3801 };
//   for (int i = 0; i < num_rollouts; i++)
//   {
//     assert_float_array_eq(&state_der_cpu[i * DYNAMICS::STATE_DIM + 3], expected_state_deriv, 4);
//   }
//   // Check CPU Method
//   std::cout << "Check CPU computeDynamics()" << std::endl;
//   Eigen::Map<DYNAMICS::state_array> initial_state_mat(initial_state_cpu);
//   Eigen::Map<DYNAMICS::control_array> control_cpu_mat(control_cpu);
//   DYNAMICS::state_array state_der_mat;
//   DYNAMICS::output_array output;
//   model.initializeDynamics(initial_state_mat, control_cpu_mat, output, 0.0, 0.02);
//   model.computeDynamics(initial_state_mat, control_cpu_mat, state_der_mat);
//   auto dyn_cpu_params = model.getParams();
//
//   assert_float_array_eq(&state_der_mat.data()[3], expected_state_deriv, 4);
//   HANDLE_ERROR(cudaStreamDestroy(stream));
// }
//
// TEST_F(LSTMDynamicsTest, CheckInitializationNetworkSpeed)
// {
//   DYNAMICS model(u_constraint, stream);
//   using micro = std::chrono::microseconds;
//
//   model.GPUSetup();
//   auto dyn_params = model.getParams();
//   dyn_params.W_fm[0] = 5.0;
//   dyn_params.copy_everything = true;
//   dyn_params.W_hidden_input.get()[0] = 13;
//   model.setParams(dyn_params);
//   ModelExposer access_cpu_model(&model);
//   const int iterations = 1000;
//   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//   for (int i = 0; i < iterations; i++)
//   {
//     dyn_params.updateInitialLSTMState();
//   }
//   std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//   std::cout << "updateInitialLSTMState() avg time for " << iterations
//             << " runs: " << std::chrono::duration_cast<micro>(end - begin).count() / iterations << " µs" <<
//             std::endl;
// }
//
// TEST_F(LSTMDynamicsTest, CompareComputeControl)
// {
//   using LSTM_DYNAMICS = DYNAMICS;
//   typedef NeuralNetModel<7, 2, 3, 6, 32, 32, 4> FF_DYNAMICS;
//   using micro = std::chrono::microseconds;
//
//   const int num_rollouts = 1920;
//   const int num_timesteps = 10;
//   const int blocksize_x = 8;
//   const int blocksize_y = 16;
//
//   typedef VanillaMPPIController<LSTM_DYNAMICS, ARStandardCost, DDPFeedback<LSTM_DYNAMICS, num_timesteps>,
//   num_timesteps,
//                                 num_rollouts, blocksize_x, blocksize_y>
//       LSTM_CONTROLLER;
//   typedef VanillaMPPIController<FF_DYNAMICS, ARStandardCost, DDPFeedback<FF_DYNAMICS, num_timesteps>, num_timesteps,
//                                 num_rollouts, blocksize_x, blocksize_y>
//       FF_CONTROLLER;
//
//   /** ========== Set up Dynamics Models ==========**/
//   LSTM_DYNAMICS LSTM_model(u_constraint, stream);
//   FF_DYNAMICS FF_model(u_constraint, stream);
//   ARStandardCost* costs = new ARStandardCost();
//
//   LSTM_model.GPUSetup();
//
//   FF_model.GPUSetup();
//
//   float dt = 0.02;
//   int max_iter = 1;
//   float lambda = 6.66;
//   float alpha = 0.0;
//
//   LSTM_CONTROLLER::control_array control_std_dev = 0.5 * LSTM_CONTROLLER::control_array::Ones();
//
//   auto ff_fb_controller = DDPFeedback<FF_DYNAMICS, num_timesteps>(&FF_model, dt);
//   auto lstm_fb_controller = DDPFeedback<LSTM_DYNAMICS, num_timesteps>(&LSTM_model, dt);
//
//   LSTM_CONTROLLER lstm_controller(&LSTM_model, costs, &lstm_fb_controller, dt, max_iter, lambda, alpha,
//                                   control_std_dev);
//   FF_CONTROLLER ff_controller(&FF_model, costs, &ff_fb_controller, dt, max_iter, lambda, alpha, control_std_dev);
//   lstm_controller.setCUDAStream(stream);
//   ff_controller.setCUDAStream(stream);
//
//   // Load networks and maps
//   costs->loadTrackData(mppi::tests::standard_test_map_file);
//   FF_model.loadParams(mppi::tests::old_autorally_network_file);
//   LSTM_model.loadParams(mppi::tests::autorally_lstm_network_file, mppi::tests::autorally_hidden_network_file,
//                         mppi::tests::autorally_cell_network_file, mppi::tests::autorally_output_network_file);
//
//   std::cout << "LSTM Num params: " << LSTM_model.getParams().LSTM_NUM_WEIGHTS << std::endl;
//   // std::cout << "FF Num params: " << FF_model.getParams().NUM_PARAMS << std::endl;
//
//   std::vector<float> x_0 = { 2.9642e-04, 5.7054e+00, 1.1859e-03, 1.3721e-01, 2.4944e-02, 1.2798e-01 };
//   std::vector<float> x_1 = { 7.8346e-04, 5.6928e+00, -1.4520e-02, 1.7258e-01, -3.1522e-03, 8.4512e-02 };
//   std::vector<float> x_2 = { 7.5389e-04, 5.6884e+00, -1.9062e-02, 4.5813e-04, -2.3523e-02, 6.8172e-02 };
//   std::vector<float> x_3 = { 1.5670e-03, 5.6779e+00, 5.7993e-03, -9.1165e-02, -2.5202e-02, 9.0036e-02 };
//   std::vector<float> x_4 = { 2.0307e-03, 5.6623e+00, 3.5971e-02, -1.4233e-01, -1.4520e-02, 1.3751e-01 };
//   std::vector<float> x_5 = { 6.6427e-04, 5.6565e+00, 4.3000e-02, -2.1955e-02, -2.1740e-03, 1.9203e-01 };
//   std::vector<float> x_6 = { 2.1942e-04, 5.6636e+00, 1.6840e-02, -5.7120e-03, 3.3988e-03, 2.3751e-01 };
//   std::vector<float> x_7 = { 6.9824e-04, 5.6656e+00, -4.1707e-04, -4.3693e-02, 2.9118e-03, 2.6795e-01 };
//   std::vector<float> x_8 = { 1.2957e-03, 5.6861e+00, 2.8441e-03, -1.1037e-01, -6.0677e-04, 2.7991e-01 };
//   std::vector<float> x_9 = { 8.7452e-04, 5.7010e+00, 2.2052e-02, -2.5667e-02, -4.2457e-03, 2.7212e-01 };
//   std::vector<float> x_10 = { 7.2980e-04, 5.7185e+00, 2.0501e-02, -4.2951e-02, -5.5691e-03, 2.5228e-01 };
//
//   Eigen::Matrix<float, 9, 11> new_buffer;
//   // set to be random so there should be an issue if we incorrectly index
//   new_buffer.setRandom();
//
//   new_buffer.block<4, 1>(3, 0) = Eigen::Matrix<float, 4, 1>(x_0.data());
//   new_buffer.block<2, 1>(7, 0) = Eigen::Matrix<float, 2, 1>(x_0.data() + 4);
//
//   new_buffer.block<4, 1>(3, 1) = Eigen::Matrix<float, 4, 1>(x_1.data());
//   new_buffer.block<2, 1>(7, 1) = Eigen::Matrix<float, 2, 1>(x_1.data() + 4);
//
//   new_buffer.block<4, 1>(3, 2) = Eigen::Matrix<float, 4, 1>(x_2.data());
//   new_buffer.block<2, 1>(7, 2) = Eigen::Matrix<float, 2, 1>(x_2.data() + 4);
//
//   new_buffer.block<4, 1>(3, 3) = Eigen::Matrix<float, 4, 1>(x_3.data());
//   new_buffer.block<2, 1>(7, 3) = Eigen::Matrix<float, 2, 1>(x_3.data() + 4);
//
//   new_buffer.block<4, 1>(3, 4) = Eigen::Matrix<float, 4, 1>(x_4.data());
//   new_buffer.block<2, 1>(7, 4) = Eigen::Matrix<float, 2, 1>(x_4.data() + 4);
//
//   new_buffer.block<4, 1>(3, 5) = Eigen::Matrix<float, 4, 1>(x_5.data());
//   new_buffer.block<2, 1>(7, 5) = Eigen::Matrix<float, 2, 1>(x_5.data() + 4);
//
//   new_buffer.block<4, 1>(3, 6) = Eigen::Matrix<float, 4, 1>(x_6.data());
//   new_buffer.block<2, 1>(7, 6) = Eigen::Matrix<float, 2, 1>(x_6.data() + 4);
//
//   new_buffer.block<4, 1>(3, 7) = Eigen::Matrix<float, 4, 1>(x_7.data());
//   new_buffer.block<2, 1>(7, 7) = Eigen::Matrix<float, 2, 1>(x_7.data() + 4);
//
//   new_buffer.block<4, 1>(3, 8) = Eigen::Matrix<float, 4, 1>(x_8.data());
//   new_buffer.block<2, 1>(7, 8) = Eigen::Matrix<float, 2, 1>(x_8.data() + 4);
//
//   new_buffer.block<4, 1>(3, 9) = Eigen::Matrix<float, 4, 1>(x_9.data());
//   new_buffer.block<2, 1>(7, 9) = Eigen::Matrix<float, 2, 1>(x_9.data() + 4);
//
//   new_buffer.block<4, 1>(3, 10) = Eigen::Matrix<float, 4, 1>(x_10.data());
//   new_buffer.block<2, 1>(7, 10) = Eigen::Matrix<float, 2, 1>(x_10.data() + 4);
//
//   auto params = LSTM_model.getParams();
//   params.updateBuffer(new_buffer);
//   LSTM_model.setParams(params);
//
//   LSTM_model.paramsToDevice();
//
//   LSTM_CONTROLLER::state_array initial_state = LSTM_CONTROLLER::state_array::Zero();
//
//   const int iterations = 1000;
//   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//   for (int i = 0; i < iterations; i++)
//   {
//     lstm_controller.computeControl(initial_state, 1);
//     LSTM_CONTROLLER::control_trajectory control = lstm_controller.getControlSeq();
//     EXPECT_TRUE(control.allFinite());
//     EXPECT_TRUE(lstm_controller.getTargetStateSeq().allFinite());
//     if (!control.allFinite() && !lstm_controller.getTargetStateSeq().allFinite())
//     {
//       exit(-1);
//     }
//   }
//   std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//   std::cout << "LSTM dynamics in computeControl avg time for " << iterations
//             << " runs: " << std::chrono::duration_cast<micro>(end - begin).count() / iterations << " µs" <<
//             std::endl;
//
//   begin = std::chrono::steady_clock::now();
//   for (int i = 0; i < iterations; i++)
//   {
//     ff_controller.computeControl(initial_state, 1);
//   }
//   end = std::chrono::steady_clock::now();
//   std::cout << "FF dynamics in computeControl avg time for " << iterations
//             << " runs: " << std::chrono::duration_cast<micro>(end - begin).count() / iterations << " µs" <<
//             std::endl;
// }
