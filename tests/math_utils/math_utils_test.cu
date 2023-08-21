#include <array>

#include <gtest/gtest.h>
#include <mppi/utils/test_helper.h>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/gpu_err_chk.cuh>
#include <mppi/utils/cuda_math_utils.cuh>

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

TEST(MATH_UTILS, RotatePointByQuat)
{
  for (int i = 0; i < 100; i++)
  {
    Eigen::Quaternionf rotation = Eigen::Quaternionf::UnitRandom();
    Eigen::Vector3f translation = Eigen::Vector3f::Random();

    float3 point = make_float3(translation.x(), translation.y(), translation.z());
    float q[4] = { rotation.w(), rotation.x(), rotation.y(), rotation.z() };
    mppi::math::RotatePointByQuat(q, point);

    auto result = rotation * translation;

    EXPECT_NEAR(point.x, result.x(), 1.0e-5);
    EXPECT_NEAR(point.y, result.y(), 1.0e-5);
    EXPECT_NEAR(point.z, result.z(), 1.0e-5);
  }
}

TEST(MATH_UTILS, QuatDCM)
{
  for (int iteration = 0; iteration < 100; iteration++)
  {
    Eigen::Quaternionf q_eig = Eigen::Quaternionf::UnitRandom();
    float q[4] = { q_eig.w(), q_eig.x(), q_eig.y(), q_eig.z() };
    float M[3][3];
    Eigen::Matrix3f M_eig;
    M_eig = q_eig.toRotationMatrix();
    mppi::math::Quat2DCM(q, M);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(M_eig(i, j), M[i][j], 1.0e-5);
      }
    }
  }
}

TEST(MATH_UTILS, EulerDCM)
{
  for (int iteration = 0; iteration < 100; iteration++)
  {
    Eigen::Quaternionf q_eig = Eigen::Quaternionf::UnitRandom();
    float r, p, y;
    float M[3][3];
    Eigen::Matrix3f M_eig;
    M_eig = q_eig.toRotationMatrix();

    mppi::math::Quat2EulerNWU(q_eig, r, p, y);
    mppi::math::Euler2DCM_NWU(r, p, y, M);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(M_eig(i, j), M[i][j], 1.0e-5);
      }
    }
  }
}

TEST(MATH_UTILS, SkewSymmetricMatrixSameAsCrossProd)
{
  Eigen::Vector3f a(1, 2, 3);
  Eigen::Vector3f b(8, 3, 9);
  eigen_assert_float_eq<Eigen::Vector3f>(a.cross(b), mppi::math::skewSymmetricMatrix(a) * b);
}

namespace mm = mppi::matrix_multiplication;
namespace mp1 = mppi::p1;
namespace mp2 = mppi::p2;

template <int M, int K, int N, mp1::Parallel1Dir PARALLELIZATION_DIR, mm::MAT_OP A_OP = mm::MAT_OP::NONE,
          mm::MAT_OP B_OP = mm::MAT_OP::NONE, class T = float>
__global__ void gemm1d(T* A, T* B, T* C, T alpha = 1, T beta = 0, int shared_mem_size = 0)
{
  extern __shared__ float buff[];
  T* A_p;
  T* B_p;
  if (shared_mem_size != 0)
  {
    A_p = (T*)&buff[0];
    B_p = &A_p[M * K];
    mp1::loadArrayParallel<M * K, PARALLELIZATION_DIR, T>(A_p, 0, A, 0);
    mp1::loadArrayParallel<K * N, PARALLELIZATION_DIR, T>((T*)&buff[0], M * K, B, 0);
    __syncthreads();
  }
  else
  {
    A_p = A;
    B_p = B;
  }

  mm::gemm1<M, K, N, PARALLELIZATION_DIR, T>(A_p, B_p, C, alpha, beta, A_OP, B_OP);
}

template <int M, int K, int N>
__global__ void gemm1dBatchKernel(const float* A, const float* B, float* C)
{
  const float* A_local = &A[M * K * threadIdx.x];
  const float* B_local = &B[K * N * threadIdx.x];
  float* C_local = &C[M * N * threadIdx.x];
  mm::gemm1<M, K, N, mp1::Parallel1Dir::THREAD_Y>(A_local, B_local, C_local);
}

template <int M, int K, int N, mm::MAT_OP A_OP, mm::MAT_OP B_OP, mp1::Parallel1Dir P_DIR, class T = float>
float2 testMatMulHost(int size, T alpha = 1, T beta = 0)
{
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_MAT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_MAT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C_MAT;

  A_MAT A;
  B_MAT B;
  C_MAT C_Eigen = C_MAT::Ones(M, N);
  C_MAT C_ours = C_Eigen;
  C_Eigen *= beta;
  float eigen_time_ms = 0;

  if (A_OP == mm::MAT_OP::NONE && B_OP == mm::MAT_OP::NONE)
  {
    A = A_MAT::Random(M, K);
    B = B_MAT::Random(K, N);
    auto start = std::chrono::steady_clock::now();
    C_Eigen += A * B * alpha;
    auto stop = std::chrono::steady_clock::now();
    eigen_time_ms = mppi::math::timeDiffms(stop, start);
  }
  else if (A_OP == mm::MAT_OP::TRANSPOSE && B_OP == mm::MAT_OP::NONE)
  {
    A = A_MAT::Random(K, M);
    B = B_MAT::Random(K, N);
    auto start = std::chrono::steady_clock::now();
    C_Eigen += A.transpose() * B * alpha;
    auto stop = std::chrono::steady_clock::now();
    eigen_time_ms = mppi::math::timeDiffms(stop, start);
  }
  if (A_OP == mm::MAT_OP::NONE && B_OP == mm::MAT_OP::TRANSPOSE)
  {
    A = A_MAT::Random(M, K);
    B = B_MAT::Random(N, K);
    auto start = std::chrono::steady_clock::now();
    C_Eigen += A * B.transpose() * alpha;
    auto stop = std::chrono::steady_clock::now();
    eigen_time_ms = mppi::math::timeDiffms(stop, start);
  }
  else if (A_OP == mm::MAT_OP::TRANSPOSE && B_OP == mm::MAT_OP::TRANSPOSE)
  {
    A = A_MAT::Random(K, M);
    B = B_MAT::Random(N, K);
    auto start = std::chrono::steady_clock::now();
    C_Eigen += A.transpose() * B.transpose() * alpha;
    auto stop = std::chrono::steady_clock::now();
    eigen_time_ms = mppi::math::timeDiffms(stop, start);
  }

  auto our_start = std::chrono::steady_clock::now();
  mm::gemm1<M, K, N, P_DIR, T>(A.data(), B.data(), C_ours.data(), alpha, beta, A_OP, B_OP);
  auto our_stop = std::chrono::steady_clock::now();
  float ours_time_ms = mppi::math::timeDiffms(our_stop, our_start);
  double avg_err = (C_Eigen - C_ours).norm() / C_Eigen.size();
  EXPECT_TRUE(avg_err < 1e-7) << "C_Eigen:\n" << C_Eigen << "\nC_ours:\n" << C_ours;
  return make_float2(eigen_time_ms, ours_time_ms);
}

template <int M, int K, int N, mm::MAT_OP A_OP, mm::MAT_OP B_OP, mp1::Parallel1Dir P_DIR, class T = float>
float2 testMatMulp1(int size, T alpha = 1, T beta = 0)
{
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_MAT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_MAT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C_MAT;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  A_MAT A;
  B_MAT B;
  C_MAT C_CPU = C_MAT::Ones(M, N);
  C_MAT C_GPU = C_CPU;
  C_CPU *= beta;

  // Device Memory Locations
  T* A_d;
  T* B_d;
  T* C_d;
  dim3 block(1, 1, 1);
  dim3 grid(1, 1, 1);

  HANDLE_ERROR(cudaMalloc((void**)&A_d, sizeof(T) * M * K));
  HANDLE_ERROR(cudaMalloc((void**)&B_d, sizeof(T) * K * N));
  HANDLE_ERROR(cudaMalloc((void**)&C_d, sizeof(T) * M * N));
  float cpu_time_ms = 0;

  if (A_OP == mm::MAT_OP::NONE && B_OP == mm::MAT_OP::NONE)
  {
    A = A_MAT::Random(M, K);
    B = B_MAT::Random(K, N);
    auto start = std::chrono::steady_clock::now();
    C_CPU += A * B * alpha;
    auto stop = std::chrono::steady_clock::now();
    cpu_time_ms = mppi::math::timeDiffms(stop, start);
  }
  else if (A_OP == mm::MAT_OP::TRANSPOSE && B_OP == mm::MAT_OP::NONE)
  {
    A = A_MAT::Random(K, M);
    B = B_MAT::Random(K, N);
    auto start = std::chrono::steady_clock::now();
    C_CPU += A.transpose() * B * alpha;
    auto stop = std::chrono::steady_clock::now();
    cpu_time_ms = mppi::math::timeDiffms(stop, start);
  }
  if (A_OP == mm::MAT_OP::NONE && B_OP == mm::MAT_OP::TRANSPOSE)
  {
    A = A_MAT::Random(M, K);
    B = B_MAT::Random(N, K);
    auto start = std::chrono::steady_clock::now();
    C_CPU += A * B.transpose() * alpha;
    auto stop = std::chrono::steady_clock::now();
    cpu_time_ms = mppi::math::timeDiffms(stop, start);
  }
  else if (A_OP == mm::MAT_OP::TRANSPOSE && B_OP == mm::MAT_OP::TRANSPOSE)
  {
    A = A_MAT::Random(K, M);
    B = B_MAT::Random(N, K);
    auto start = std::chrono::steady_clock::now();
    C_CPU += A.transpose() * B.transpose() * alpha;
    auto stop = std::chrono::steady_clock::now();
    cpu_time_ms = mppi::math::timeDiffms(stop, start);
  }

  int grid_dim = 1;

  if (P_DIR == mp1::Parallel1Dir::THREAD_X)
  {
    block.x = size;
  }
  else if (P_DIR == mp1::Parallel1Dir::THREAD_Y)
  {
    block.y = size;
  }
  else if (P_DIR == mp1::Parallel1Dir::THREAD_Z)
  {
    block.z = size;
  }
  else if (P_DIR == mp1::Parallel1Dir::GLOBAL_X)
  {
    grid_dim = (M * N - 1) / (size) + 1;
    block.x = size;
    grid.x = grid_dim;
  }
  else if (P_DIR == mp1::Parallel1Dir::GLOBAL_Y)
  {
    grid_dim = (M * N - 1) / (size) + 1;
    block.y = size;
    grid.y = grid_dim;
  }
  else if (P_DIR == mp1::Parallel1Dir::GLOBAL_Z)
  {
    grid_dim = (M * N - 1) / (size) + 1;
    block.y = size;
    grid.y = grid_dim;
  }
  int shared_mem_num = ((M * K + K * N) * sizeof(T) <= 1 << 16) ? (M * K + K * N) : 0;

  HANDLE_ERROR(cudaMemcpyAsync(A_d, A.data(), sizeof(T) * M * K, cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(B_d, B.data(), sizeof(T) * K * N, cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(C_d, C_GPU.data(), sizeof(T) * M * N, cudaMemcpyHostToDevice, stream));
  cudaEventRecord(start, stream);
  gemm1d<M, K, N, P_DIR, A_OP, B_OP, T>
      <<<grid, block, shared_mem_num * sizeof(T), stream>>>(A_d, B_d, C_d, alpha, beta, shared_mem_num);
  cudaEventRecord(stop, stream);
  HANDLE_ERROR(cudaMemcpyAsync(C_GPU.data(), C_d, sizeof(T) * M * N, cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaEventSynchronize(stop));

  float gpu_time_ms = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));

  double avg_err = (C_CPU - C_GPU).norm() / C_CPU.size();
  EXPECT_TRUE(avg_err < 1e-7) << "C_CPU:\n" << C_CPU << "\nC_GPU:\n" << C_GPU;
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  return make_float2(cpu_time_ms, gpu_time_ms);
};

TEST(MATH_UTILS, float_BasicMatrixMult_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<10, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<10, 2, 50, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(256, 1, 0);
}

TEST(MATH_UTILS, float_BasicMatrixMult_y)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Y, float>(16, 1, 0);
  testMatMulp1<10, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Y, float>(16, 1, 0);
  testMatMulp1<10, 2, 50, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Y, float>(256, 1, 0);
}

TEST(MATH_UTILS, float_BasicMatrixMult_z)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Z, float>(16, 1, 0);
  testMatMulp1<10, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Z, float>(16, 1, 0);
  testMatMulp1<10, 2, 50, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Z, float>(64, 1, 0);
}

TEST(MATH_UTILS, float_BasicMatrixMult_none)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::NONE, float>(16, 1, 0);
  testMatMulp1<10, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::NONE, float>(16, 1, 0);
  testMatMulp1<10, 2, 50, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::NONE, float>(256, 1, 0);
}

TEST(MATH_UTILS, float_TransposeA_MatrixMult_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
}

TEST(MATH_UTILS, float_TransposeA_TransposeB_MatrixMult_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
}

TEST(MATH_UTILS, float_TransposeB_MatrixMult_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
}

TEST(MATH_UTILS, int_BasicMatrixMult_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Z, int>(16, 1, 0);
}

TEST(MATH_UTILS, double_BasicMatrixMult_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Z, double>(16, 1, 0);
}

TEST(MATH_UTILS, float_MatrixMult_UnalignedK_x)
{
  testMatMulp1<3, 17, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 17, 5, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 17, 5, mm::MAT_OP::NONE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 1, 4, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
}

TEST(MATH_UTILS, float_MatrixMult_alignedK2_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 2, 5, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 2, 5, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
}

TEST(MATH_UTILS, float_MatrixMult_alignedK4_x)
{
  testMatMulp1<3, 4, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 4, 5, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 4, 5, mm::MAT_OP::NONE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
  testMatMulp1<3, 4, 5, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::TRANSPOSE, mp1::Parallel1Dir::THREAD_X, float>(16, 1, 0);
}

TEST(MATH_UTILS, float_BasicMatrixMult_alpha_z)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Z, float>(16, 3.0, 0);
}

TEST(MATH_UTILS, float_BasicMatrixMult_alpha_beta_z)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_Z, float>(16, 3.0, 0.2);
}

TEST(MATH_UTILS, float_BasicMatrixMult_alpha_beta_x)
{
  testMatMulp1<3, 2, 5, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(5, 3.0, 0.2);
}

TEST(MATH_UTILS, HostMatrixMult)
{
  const int M = 10;
  const int K = 5;
  const int N = 6;
  const mm::MAT_OP A_OP = mm::MAT_OP::NONE;
  const mm::MAT_OP B_OP = mm::MAT_OP::NONE;
  const mp1::Parallel1Dir P_DIR = mp1::Parallel1Dir::THREAD_X;
  using T = float;
  T alpha = 1.0;
  T beta = 0.0;
  testMatMulHost<M, K, N, A_OP, B_OP, P_DIR, T>(alpha, beta);
}
TEST(MATH_UTILS, SpeedSharedMemMatrixMult)
{
  const int NUM_ITERATIONS = 100;
  float2 total_ms = make_float2(0, 0);
  const int thread_dim = 1024;
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    total_ms +=
        testMatMulp1<128, 256, 128, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(
            thread_dim, 3.0, 0.2);
  }
  std::cout << NUM_ITERATIONS << " GPU Shared Mem Matrix Multiplications took " << total_ms.y / NUM_ITERATIONS
            << " ms on average" << std::endl;
  std::cout << NUM_ITERATIONS << " CPU Shared Mem Matrix Multiplications took " << total_ms.x / NUM_ITERATIONS
            << " ms on average" << std::endl;
}

TEST(MATH_UTILS, SpeedLargeMatrixMult)
{
  const int NUM_ITERATIONS = 5;
  float2 total_ms = make_float2(0, 0);
  const int thread_dim = 1024;
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    total_ms +=
        testMatMulp1<1024, 2048, 1024, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(
            thread_dim, 3.0, 0.2);
  }
  std::cout << NUM_ITERATIONS << " GPU Matrix Multiplications took " << total_ms.y / NUM_ITERATIONS << " ms on average"
            << std::endl;
  std::cout << NUM_ITERATIONS << " CPU Matrix Multiplications took " << total_ms.x / NUM_ITERATIONS << " ms on average"
            << std::endl;
}

TEST(MATH_UTILS, SpeedLargeMatrixMultGlobal)
{
  const int NUM_ITERATIONS = 10;
  float2 total_ms = make_float2(0, 0);
  const int thread_dim = 1024;
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    total_ms +=
        testMatMulp1<1024, 2048, 1024, mm::MAT_OP::TRANSPOSE, mm::MAT_OP::NONE, mp1::Parallel1Dir::GLOBAL_X, float>(
            thread_dim, 3.0, 0.2);
  }
  std::cout << NUM_ITERATIONS << " GPU Matrix Multiplications took " << total_ms.y / NUM_ITERATIONS << " ms on average"
            << std::endl;
  std::cout << NUM_ITERATIONS << " CPU Matrix Multiplications took " << total_ms.x / NUM_ITERATIONS << " ms on average"
            << std::endl;
}

TEST(MATH_UTILS, SpeedSmallMatrixMult)
{
  const int NUM_ITERATIONS = 10;
  float2 total_ms = make_float2(0, 0);
  const int thread_dim = 16;
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    total_ms += testMatMulp1<5, 4, 3, mm::MAT_OP::NONE, mm::MAT_OP::NONE, mp1::Parallel1Dir::THREAD_X, float>(
        thread_dim, 3.0, 0.2);
  }
  std::cout << NUM_ITERATIONS << " GPU Matrix Multiplications took " << total_ms.y / NUM_ITERATIONS << " ms on average"
            << std::endl;
  std::cout << NUM_ITERATIONS << " CPU Matrix Multiplications took " << total_ms.x / NUM_ITERATIONS << " ms on average"
            << std::endl;
}

/** 2 Dimensional Parallelization Tests **/

template <int M, int K, int N, mp2::Parallel2Dir PARALLELIZATION_DIR>
__global__ void gemm2d(const float* A, const float* B, float* C)
{
  mm::gemm2<M, K, N, PARALLELIZATION_DIR>(A, B, C);
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

  auto inner_loop_before = [&](int size1, int size2, mp2::Parallel2Dir P_DIR) {
    A = A_mat::Random();
    B = B_mat::Random();
    C_CPU = A * B;
    block = dim3(1, 1, 1);
    if (P_DIR == mp2::Parallel2Dir::THREAD_XY)
    {
      block.x = size1;
      block.y = max(size2, 1);
    }
    else if (P_DIR == mp2::Parallel2Dir::THREAD_XZ)
    {
      block.x = size1;
      block.z = max(min(size2, 64), 1);
    }
    else if (P_DIR == mp2::Parallel2Dir::THREAD_YX)
    {
      block.y = size1;
      block.x = max(size2, 1);
    }
    else if (P_DIR == mp2::Parallel2Dir::THREAD_YZ)
    {
      block.y = size1;
      block.z = max(min(size2, 64), 1);
    }
    else if (P_DIR == mp2::Parallel2Dir::THREAD_ZY)
    {
      block.z = max(min(size1, 64), 1);
      block.y = max(size2, 1);
    }
    else if (P_DIR == mp2::Parallel2Dir::THREAD_ZX)
    {
      block.z = max(min(size1, 64), 1);
      block.x = max(size2, 1);
    }
    cudaEventRecord(start, stream);
    HANDLE_ERROR(cudaMemcpyAsync(A_d, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(B_d, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice, stream));
  };

  auto inner_loop_after = [&](int size1, int size2, mp2::Parallel2Dir P_DIR) {
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
    // if (P_DIR == mp2::Parallel2Dir::THREAD_XY)
    // {
    //   dir_s = "xy";
    // } else if (P_DIR == mp2::Parallel2Dir::THREAD_XZ)
    // {
    //   dir_s = "xz";
    // } else if (P_DIR == mp2::Parallel2Dir::THREAD_YX)
    // {
    //   dir_s = "yx";
    // } else if (P_DIR == mp2::Parallel2Dir::THREAD_YZ)
    // {
    //   dir_s = "yz";
    // } else if (P_DIR == mp2::Parallel2Dir::THREAD_ZY)
    // {
    //   dir_s = "zy";
    // } else if (P_DIR == mp2::Parallel2Dir::THREAD_ZX)
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
    inner_loop_before(block_size_1, block_size_2, mp2::Parallel2Dir::THREAD_XY);
    gemm2d<M, K, N, mp2::Parallel2Dir::THREAD_XY><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
    inner_loop_after(block_size_1, block_size_2, mp2::Parallel2Dir::THREAD_XY);
  }
  for (int block_size_1 = 1; block_size_1 <= max_thread_size; block_size_1++)
  {
    int block_size_2 = max_thread_size - block_size_1;
    inner_loop_before(block_size_1, block_size_2, mp2::Parallel2Dir::THREAD_XZ);
    gemm2d<M, K, N, mp2::Parallel2Dir::THREAD_XZ><<<grid, block, 0, stream>>>(A_d, B_d, C_d);
    inner_loop_after(block_size_1, block_size_2, mp2::Parallel2Dir::THREAD_XZ);
  }
  for (int block_size_1 = 1; block_size_1 <= max_thread_size; block_size_1++)
  {
    int block_size_2 = max_thread_size - block_size_1;
    inner_loop_before(block_size_1, block_size_2, mp2::Parallel2Dir::THREAD_YZ);
    gemm2d<M, K, N, mp2::Parallel2Dir::THREAD_YZ><<<grid, block, 0, stream>>>(A_d, B_d, C_d);

    inner_loop_after(block_size_1, block_size_2, mp2::Parallel2Dir::THREAD_YZ);
    // std::cout << "C_GPU:\n" << C_GPU << "\nC_CPU:\n" << C_CPU << std::endl;
  }

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
