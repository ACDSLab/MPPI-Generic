#include <array>

#include <gtest/gtest.h>
#include <mppi/utils/test_helper.h>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/gpu_err_chk.cuh>

TEST(MATH_UTILS, QuatInv)
{
  // Create an unnormalized quaternion.
  std::array<float, 4> q{ 1, 2, 3, 2 };
  std::array<float, 4> q_inv{};

  mppi::math::QuatInv(q.data(), q_inv.data());

  // Compute the inverse using eigen.
  // Don't use the array constructor because eigen stores it as [x, y, z, w] internally.
  const Eigen::Quaternionf eigen_q{ q[0], q[1], q[2], q[3] };
  const Eigen::Quaternionf eigen_q_inv = eigen_q.inverse().normalized();
  std::array<float, 4> correct_q_inv{ eigen_q_inv.w(), eigen_q_inv.x(), eigen_q_inv.y(), eigen_q_inv.z() };

  // q_inv should be normalized.
  constexpr float tol = 1e-7;
  array_assert_float_near<4>(q_inv, correct_q_inv, tol);
}

TEST(MATH_UTILS, QuatMultiply)
{
  // Create an unnormalized quaternion.
  std::array<float, 4> q1{ 1, 2, 3, 4 };
  std::array<float, 4> q2{ 8, 7, 6, 5 };
  std::array<float, 4> q3{};

  mppi::math::QuatMultiply(q1.data(), q2.data(), q3.data());

  // Compare the multiplication using eigen.
  // Don't use the array constructor because eigen stores it as [x, y, z, w] internally.
  const Eigen::Quaternionf eigen_q1{ q1[0], q1[1], q1[2], q1[3] };
  const Eigen::Quaternionf eigen_q2{ q2[0], q2[1], q2[2], q2[3] };
  const Eigen::Quaternionf eigen_q3 = (eigen_q1 * eigen_q2).normalized();

  std::array<float, 4> correct_q3{ eigen_q3.w(), eigen_q3.x(), eigen_q3.y(), eigen_q3.z() };

  // q_inv should be normalized.
  constexpr auto tol = 1e-7;
  array_assert_float_near<4>(q3, correct_q3, tol);
}

TEST(MATH_UTILS, SkewSymmetricMatrixSameAsCrossProd)
{
  Eigen::Vector3f a(1, 2, 3);
  Eigen::Vector3f b(8, 3, 9);
  eigen_assert_float_eq<Eigen::Vector3f>(a.cross(b), mppi::math::skewSymmetricMatrix(a) * b);
}

namespace mm1 = mppi::matrix_multiplication::p1;
namespace mm2 = mppi::matrix_multiplication::p2;

template <int M, int K, int N, mm1::Parallel1Dir PARALLELIZATION_DIR>
__global__ void gemm1d(const float* A, const float* B, float* C)
{
  mm1::gemm<M, K, N, PARALLELIZATION_DIR>(A, B, C);
}

template <int M, int K, int N>
__global__ void gemm1dBatchKernel(const float* A, const float* B, float* C)
{
  const float* A_local = &A[M * K * threadIdx.x];
  const float* B_local = &B[K * N * threadIdx.x];
  float* C_local = &C[M * N * threadIdx.x];
  mm1::gemm<M, K, N, mm1::Parallel1Dir::THREAD_Y>(A_local, B_local, C_local);
}

TEST(MATH_UTILS, Matrix3fMultiplicationP1)
{
  const int M = 300;
  const int K = 3;
  const int N = 100;
  const int max_thread_size = 30;
  // typedef Eigen::Matrix<float, M, K> A_mat;
  // typedef Eigen::Matrix<float, K, N> B_mat;
  // typedef Eigen::Matrix<float, M, N> C_mat;

  typedef Eigen::MatrixXf A_mat;
  typedef Eigen::MatrixXf B_mat;
  typedef Eigen::MatrixXf C_mat;

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  A_mat A;
  B_mat B;
  C_mat C_CPU;
  C_mat C_GPU;
  // std::cout << "C_CPU:\n" << C_CPU << std::endl;
  float* A_d;
  float* B_d;
  float* C_d;
  dim3 block(1, 1, 1);
  dim3 grid(1, 1, 1);

  HANDLE_ERROR(cudaMalloc((void**)&A_d, sizeof(float) * M * K));
  HANDLE_ERROR(cudaMalloc((void**)&B_d, sizeof(float) * K * N));
  HANDLE_ERROR(cudaMalloc((void**)&C_d, sizeof(float) * M * N));

  auto inner_loop_before = [&](int size, mm1::Parallel1Dir P_DIR) {
    // A = A_mat::Random();
    // B = B_mat::Random();
    A = Eigen::MatrixXf::Random(M, K);
    B = Eigen::MatrixXf::Random(K, N);
    C_CPU = A * B;
    block = dim3(1, 1, 1);
    if (P_DIR == mm1::Parallel1Dir::THREAD_X)
    {
      block.x = size;
    }
    else if (P_DIR == mm1::Parallel1Dir::THREAD_Y)
    {
      block.y = size;
    }
    else if (P_DIR == mm1::Parallel1Dir::THREAD_Z)
    {
      block.z = size;
    }
    cudaEventRecord(start, stream);
    HANDLE_ERROR(cudaMemcpyAsync(A_d, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(B_d, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice, stream));
  };

  auto inner_loop_after = [&](int size, mm1::Parallel1Dir P_DIR) {
    HANDLE_ERROR(cudaMemcpyAsync(C_GPU.data(), C_d, sizeof(float) * C_GPU.size(), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    cudaEventRecord(stop, stream);
    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    // std::cout << "C_GPU " << " with " << block_x_size << " parallelization:\n" << C_GPU << std::endl;
    // std::cout << block_x_size << " parallelization diff " << duration << " ms:\n" << C_CPU - C_GPU << std::endl;
    double avg_err = (C_CPU - C_GPU).norm() / C_CPU.size();
    // std::cout << size << " parallelization diff " << duration << " ms: " << avg_err << std::endl;
    std::string dir_s = "none";
    if (P_DIR == mm1::Parallel1Dir::THREAD_X)
    {
      dir_s = "x";
    }
    else if (P_DIR == mm1::Parallel1Dir::THREAD_Y)
    {
      dir_s = "y";
    }
    else if (P_DIR == mm1::Parallel1Dir::THREAD_Z)
    {
      dir_s = "z";
    }
    printf("%4d parallelization in %s-dir, time elapsed %e ms, diff: %f\n", size, dir_s.c_str(), duration, avg_err);
    ASSERT_TRUE(avg_err < 1e-07);
  };

  for (int block_x_size = 1; block_x_size <= max_thread_size; block_x_size++)
  {
    inner_loop_before(block_x_size, mm1::Parallel1Dir::THREAD_X);
    gemm1d<M, K, N, mm1::Parallel1Dir::THREAD_X><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
    inner_loop_after(block_x_size, mm1::Parallel1Dir::THREAD_X);
  }

  for (int block_y_size = 1; block_y_size <= max_thread_size; block_y_size++)
  {
    inner_loop_before(block_y_size, mm1::Parallel1Dir::THREAD_Y);
    gemm1d<M, K, N, mm1::Parallel1Dir::THREAD_Y><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
    inner_loop_after(block_y_size, mm1::Parallel1Dir::THREAD_Y);
  }

  for (int block_z_size = 1; block_z_size <= 64; block_z_size++)
  {
    inner_loop_before(block_z_size, mm1::Parallel1Dir::THREAD_Z);
    gemm1d<M, K, N, mm1::Parallel1Dir::THREAD_Z><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
    inner_loop_after(block_z_size, mm1::Parallel1Dir::THREAD_Z);
  }
  inner_loop_before(1, mm1::Parallel1Dir::NONE);
  gemm1d<M, K, N, mm1::Parallel1Dir::NONE><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
  inner_loop_after(1, mm1::Parallel1Dir::NONE);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

template <int M, int K, int N, mm2::Parallel2Dir PARALLELIZATION_DIR>
__global__ void gemm2d(const float* A, const float* B, float* C)
{
  mm2::gemm<M, K, N, PARALLELIZATION_DIR>(A, B, C);
}

TEST(MATH_UTILS, DISABLED_Matrix3fMultiplicationP2)
{
  const int M = 300;
  const int K = 30;
  const int N = 3;
  const int max_thread_size = 1;
  typedef Eigen::Matrix<float, M, K> A_mat;
  typedef Eigen::Matrix<float, K, N> B_mat;
  typedef Eigen::Matrix<float, M, N> C_mat;

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  A_mat A = A_mat::Random();
  B_mat B = B_mat::Random();
  C_mat C_CPU = A * B;
  C_mat C_GPU;
  // std::cout << "C_CPU:\n" << C_CPU << std::endl;
  float* A_d;
  float* B_d;
  float* C_d;
  dim3 block(1, 1, 1);
  dim3 grid(1, 1, 1);

  HANDLE_ERROR(cudaMalloc((void**)&A_d, sizeof(float) * A.size()));
  HANDLE_ERROR(cudaMalloc((void**)&B_d, sizeof(float) * B.size()));
  HANDLE_ERROR(cudaMalloc((void**)&C_d, sizeof(float) * C_CPU.size()));

  auto inner_loop_before = [&](int size1, int size2, mm2::Parallel2Dir P_DIR) {
    A = A_mat::Random();
    B = B_mat::Random();
    C_CPU = A * B;
    block = dim3(1, 1, 1);
    if (P_DIR == mm2::Parallel2Dir::THREAD_XY)
    {
      block.x = size1;
      block.y = max(size2, 1);
    }
    else if (P_DIR == mm2::Parallel2Dir::THREAD_XZ)
    {
      block.x = size1;
      block.z = max(min(size2, 64), 1);
    }
    else if (P_DIR == mm2::Parallel2Dir::THREAD_YX)
    {
      block.y = size1;
      block.x = max(size2, 1);
    }
    else if (P_DIR == mm2::Parallel2Dir::THREAD_YZ)
    {
      block.y = size1;
      block.z = max(min(size2, 64), 1);
    }
    else if (P_DIR == mm2::Parallel2Dir::THREAD_ZY)
    {
      block.z = max(min(size1, 64), 1);
      block.y = max(size2, 1);
    }
    else if (P_DIR == mm2::Parallel2Dir::THREAD_ZX)
    {
      block.z = max(min(size1, 64), 1);
      block.x = max(size2, 1);
    }
    cudaEventRecord(start, stream);
    HANDLE_ERROR(cudaMemcpyAsync(A_d, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(B_d, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice, stream));
  };

  auto inner_loop_after = [&](int size1, int size2, mm2::Parallel2Dir P_DIR) {
    HANDLE_ERROR(cudaMemcpyAsync(C_GPU.data(), C_d, sizeof(float) * C_GPU.size(), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    cudaEventRecord(stop, stream);
    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    // std::cout << "C_GPU " << " with " << block_x_size << " parallelization:\n" << C_GPU << std::endl;
    // std::cout << block_x_size << " parallelization diff " << duration << " ms:\n" << C_CPU - C_GPU << std::endl;
    double avg_err = (C_CPU - C_GPU).norm() / C_CPU.size();
    // std::cout << block_x_size << " parallelization diff " << duration << " ms: " << avg_err << std::endl;
    // std::string dir_s = "none";
    // if (P_DIR == mm2::Parallel2Dir::THREAD_XY)
    // {
    //   dir_s = "xy";
    // } else if (P_DIR == mm2::Parallel2Dir::THREAD_XZ)
    // {
    //   dir_s = "xz";
    // } else if (P_DIR == mm2::Parallel2Dir::THREAD_YX)
    // {
    //   dir_s = "yx";
    // } else if (P_DIR == mm2::Parallel2Dir::THREAD_YZ)
    // {
    //   dir_s = "yz";
    // } else if (P_DIR == mm2::Parallel2Dir::THREAD_ZY)
    // {
    //   dir_s = "zy";
    // } else if (P_DIR == mm2::Parallel2Dir::THREAD_ZX)
    // {
    //   dir_s = "zx";
    // }
    // printf("%d-%d-%d parallelization in %s-dir, time elapsed %f ms, diff: %f\n", block.x, block.y, block.z,
    // dir_s.c_str(), duration, avg_err);
    printf("%d-%d-%d parallelization, time elapsed %f ms, diff: %f\n", block.x, block.y, block.z, duration, avg_err);
    ASSERT_TRUE(avg_err < 1e-07);
  };

  for (int block_size_1 = 1; block_size_1 <= max_thread_size; block_size_1++)
  {
    int block_size_2 = max_thread_size - block_size_1;
    inner_loop_before(block_size_1, block_size_2, mm2::Parallel2Dir::THREAD_XY);
    gemm2d<M, K, N, mm2::Parallel2Dir::THREAD_XY><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
    inner_loop_after(block_size_1, block_size_2, mm2::Parallel2Dir::THREAD_XY);
  }
  for (int block_size_1 = 1; block_size_1 <= max_thread_size; block_size_1++)
  {
    int block_size_2 = max_thread_size - block_size_1;
    inner_loop_before(block_size_1, block_size_2, mm2::Parallel2Dir::THREAD_XZ);
    gemm2d<M, K, N, mm2::Parallel2Dir::THREAD_XZ><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
    inner_loop_after(block_size_1, block_size_2, mm2::Parallel2Dir::THREAD_XZ);
  }
  for (int block_size_1 = 1; block_size_1 <= max_thread_size; block_size_1++)
  {
    int block_size_2 = max_thread_size - block_size_1;
    inner_loop_before(block_size_1, block_size_2, mm2::Parallel2Dir::THREAD_YZ);
    gemm2d<M, K, N, mm2::Parallel2Dir::THREAD_YZ><<<grid, block, 0, stream>>>(A_d, B_d, C_d);

    inner_loop_after(block_size_1, block_size_2, mm2::Parallel2Dir::THREAD_YZ);
    // std::cout << "C_GPU:\n" << C_GPU << "\nC_CPU:\n" << C_CPU << std::endl;
  }

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
