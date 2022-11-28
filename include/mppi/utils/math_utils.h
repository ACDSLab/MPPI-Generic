/*
 * Created on Mon Jun 01 2020 by Bogdan
 *
 */
#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

// Needed for sampling without replacement
#include <chrono>
#include <type_traits>
#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <Eigen/Dense>

#ifndef SQ
#define SQ(a) a* a
#endif  // SQ

#ifndef __UNROLL
#define __xstr__(s) __str__(s)
#define __str__(s) #s
#ifdef __CUDACC__
#define __UNROLL(a) _Pragma("unroll")
#else  // GCC is the compiler and uses different unroll syntax
#define __UNROLL(a) _Pragma(__xstr__(GCC unroll a))
#endif
#endif

namespace mppi
{
namespace math
{
const float GRAVITY = 9.81;
// Based off of https://gormanalysis.com/blog/random-numbers-in-cpp
inline std::vector<int> sample_without_replacement(const int k, const int N,
                                                   std::default_random_engine g = std::default_random_engine())
{
  if (k > N)
  {
    throw std::logic_error("Can't sample more than n times without replacement");
  }
  // Create an unordered set to store the samples
  std::unordered_set<int> samples;

  // For loop runs k times
  for (int r = N - k; r < N; r++)
  {
    if (r == 0)
    {
      samples.insert(N - 1);
      continue;
    }
    int v = std::uniform_int_distribution<>(1, r)(g);  // sample between 1 and r
    if (!samples.insert(v - 1).second)
    {  // if v exists in the set
      samples.insert(r - 1);
    }
  }
  // Copy set into a vector
  std::vector<int> final_sequence(samples.begin(), samples.end());
  // Shuffle the vector to get the final sequence of sampling
  std::shuffle(final_sequence.begin(), final_sequence.end(), g);
  return final_sequence;
}

inline __host__ __device__ float expr(const float r, const float x)
{
  float mid_term = 1.0 + (r - 1.0) * x;
  return (mid_term > 0) * powf(mid_term, 1.0 / (r - 1.0));
}

/**
 * Linear interpolation
 * Given two coordinates (x_min, y_min) and (x_max, y_max)
 * And the x location of a third (x), return the y location
 * along the line between the two points
 */
inline __host__ __device__ float linInterp(const float x, const float x_min, const float x_max, const float y_min,
                                           const float y_max)
{
  return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min;
}

/**
 * Return the sign of a variable (1 for positive, -1 for negative, 0 for 0)
 **/
template <class T = float>
inline __host__ __device__ int sign(const T& a)
{
  return (a > 0) - (a < 0);
}

/**
 * Calculates the normalized distance from the centerline
 * @param r - current radius
 * @param r_in - the inside radius of a track
 * @param r_out - the outside radius of a track
 * @return norm_dist - a normalized distance away from the centerline
 * norm_dist = 0 -> on the centerline
 * norm_dist = 1 -> on one of the track boundaries inner, or outer
 */
inline __host__ __device__ float normDistFromCenter(const float r, const float r_in, const float r_out)
{
  float r_center = (r_in + r_out) / 2;
  float r_width = (r_out - r_in);
  float dist_from_center = fabsf(r - r_center);
  float norm_dist = dist_from_center / (r_width * 0.5);
  return norm_dist;
}

/**
 * Multiply two quaternions together which gives you their rotations added together
 * q_3 = q_1 x q_2
 * Inputs:
 *  q_1 - first quaternion
 *  q_2 - second quaternion
 *  q_3 - output quaternion
 */
inline __host__ __device__ void QuatMultiply(const float q_1[4], const float q_2[4], float q_3[4])
{
  q_3[0] = q_1[0] * q_2[0] - q_1[1] * q_2[1] - q_1[2] * q_2[2] - q_1[3] * q_2[3];
  q_3[1] = q_1[1] * q_2[0] + q_1[0] * q_2[1] - q_1[3] * q_2[2] + q_1[2] * q_2[3];
  q_3[2] = q_1[2] * q_2[0] + q_1[3] * q_2[1] + q_1[0] * q_2[2] - q_1[1] * q_2[3];
  q_3[3] = q_1[3] * q_2[0] - q_1[2] * q_2[1] + q_1[1] * q_2[2] + q_1[0] * q_2[3];
  float norm = sqrtf(powf(q_3[0], 2) + powf(q_3[1], 2) + powf(q_3[2], 2) + powf(q_3[3], 2));
  for (int i = 0; i < 4; i++)
  {
    q_3[i] /= norm;
  }
}

inline __host__ __device__ void QuatInv(const float q[4], float q_inv[4])
{
  float norm = sqrtf(powf(q[0], 2) + powf(q[1], 2) + powf(q[2], 2) + powf(q[3], 2));
  q_inv[0] = q[0] / norm;
  q_inv[1] = -q[1] / norm;
  q_inv[2] = -q[2] / norm;
  q_inv[3] = -q[3] / norm;
}

/**
 * Calculate the rotation required to get from q_1 to q_2
 * In Euler angles, this would be direct subraction but
 * in quaternions, it doesn't quite work that way
 */
inline __device__ void QuatSubtract(float q_1[4], float q_2[4], float q_3[4])
{
  float q_1_inv[4];
  QuatInv(q_1, q_1_inv);
  QuatMultiply(q_2, q_1_inv, q_3);
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __device__ void Euler2QuatNWU(const double& r, const double& p, const double& y, double q[4])
{
  double phi_2 = r / 2.0;
  double theta_2 = p / 2.0;
  double psi_2 = y / 2.0;
  double cos_phi_2 = cos(phi_2);
  double sin_phi_2 = sin(phi_2);
  double cos_theta_2 = cos(theta_2);
  double sin_theta_2 = sin(theta_2);
  double cos_psi_2 = cos(psi_2);
  double sin_psi_2 = sin(psi_2);

  q[0] = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2;
  q[1] = -cos_phi_2 * sin_theta_2 * sin_psi_2 + cos_theta_2 * cos_psi_2 * sin_phi_2;
  q[2] = cos_phi_2 * cos_psi_2 * sin_theta_2 + sin_phi_2 * cos_theta_2 * sin_psi_2;
  q[3] = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * cos_psi_2 * sin_theta_2;
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __device__ void Euler2QuatNWU(const float& r, const float& p, const float& y, float q[4])
{
  float phi_2 = r / 2.0;
  float theta_2 = p / 2.0;
  float psi_2 = y / 2.0;
  float cos_phi_2 = cosf(phi_2);
  float sin_phi_2 = sinf(phi_2);
  float cos_theta_2 = cosf(theta_2);
  float sin_theta_2 = sinf(theta_2);
  float cos_psi_2 = cosf(psi_2);
  float sin_psi_2 = sinf(psi_2);

  q[0] = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2;
  q[1] = -cos_phi_2 * sin_theta_2 * sin_psi_2 + cos_theta_2 * cos_psi_2 * sin_phi_2;
  q[2] = cos_phi_2 * cos_psi_2 * sin_theta_2 + sin_phi_2 * cos_theta_2 * sin_psi_2;
  q[3] = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * cos_psi_2 * sin_theta_2;
}

// (RPY rotation sequence)
/*
 * Returns an euler sequence 3-2-1 (roll pitch yaw) that when applied takes you from body to world
 */
inline __host__ __device__ void Quat2EulerNWU(const float q[4], float& r, float& p, float& y)
{
  r = atan2f(2 * q[3] * q[2] + 2 * q[0] * q[1], q[0] * q[0] + q[3] * q[3] - q[2] * q[2] - q[1] * q[1]);
  float temp = -2 * q[0] * q[2] + 2 * q[1] * q[3];
  // Clamp value between -1 and 1 to prevent NaNs
  p = -asinf(fmaxf(fminf(1, temp), -1));
  y = atan2f(2 * q[2] * q[1] + 2 * q[3] * q[0], q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);
}

inline __host__ __device__ void Quat2DCM(const float q[4], float M[3][3])
{
  M[0][0] = SQ(q[0]) + SQ(q[1]) - SQ(q[2]) - SQ(q[3]);
  M[0][1] = 2 * (q[1] * q[2] - q[0] * q[3]);
  M[0][2] = 2 * (q[1] * q[3] + q[0] * q[2]);
  M[1][0] = 2 * (q[1] * q[2] + q[0] * q[3]);
  M[1][1] = SQ(q[0]) - SQ(q[1]) + SQ(q[2]) - SQ(q[3]);
  M[1][2] = 2 * (q[2] * q[3] - q[0] * q[1]);
  M[2][0] = 2 * (q[1] * q[3] - q[0] * q[2]);
  M[2][1] = 2 * (q[2] * q[3] + q[0] * q[1]);
  M[2][2] = SQ(q[0]) - SQ(q[1]) - SQ(q[2]) + SQ(q[3]);
}

inline void QuatSubtract(const Eigen::Quaternionf& q_1, const Eigen::Quaternionf& q_2, Eigen::Quaternionf& q_3)
{
  q_3 = q_2 * q_1.inverse();
}

inline void QuatMultiply(const Eigen::Quaternionf& q_1, const Eigen::Quaternionf& q_2, Eigen::Quaternionf& q_3)
{
  q_3 = q_1 * q_2;
}

inline void QuatInv(const Eigen::Quaternionf& q, Eigen::Quaternionf& q_f)
{
  q_f = q.inverse();
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __host__ __device__ void Euler2QuatNWU(const float& r, const float& p, const float& y, Eigen::Quaternionf& q)
{
  // double psi = clamp_radians(euler.roll);
  // double theta = clamp_radians(euler.pitch);
  // double phi = clamp_radians(euler.yaw);
  float phi_2 = r / 2.0;
  float theta_2 = p / 2.0;
  float psi_2 = y / 2.0;
  float cos_phi_2 = cosf(phi_2);
  float sin_phi_2 = sinf(phi_2);
  float cos_theta_2 = cosf(theta_2);
  float sin_theta_2 = sinf(theta_2);
  float cos_psi_2 = cosf(psi_2);
  float sin_psi_2 = sinf(psi_2);

  q.w() = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2;
  q.x() = -cos_phi_2 * sin_theta_2 * sin_psi_2 + cos_theta_2 * cos_psi_2 * sin_phi_2;
  q.y() = cos_phi_2 * cos_psi_2 * sin_theta_2 + sin_phi_2 * cos_theta_2 * sin_psi_2;
  q.z() = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * cos_psi_2 * sin_theta_2;
}

// (RPY rotation sequence)
/*
 * Returns an euler sequence 3-2-1 (roll pitch yaw) that when applied takes you from body to world
 */
inline void __host__ __device__ Quat2EulerNWU(const Eigen::Quaternionf& q, float& r, float& p, float& y)
{
  r = atan2f(2 * q.z() * q.y() + 2 * q.w() * q.x(), q.w() * q.w() + q.z() * q.z() - q.y() * q.y() - q.x() * q.x());
  float temp = -2 * q.w() * q.y() + 2 * q.x() * q.z();
  p = -asinf(fmaxf(-1, fminf(temp, 1)));
  y = atan2f(2 * q.y() * q.x() + 2 * q.z() * q.w(), q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z());
}

inline void Quat2DCM(const Eigen::Quaternionf& q, Eigen::Ref<Eigen::Matrix3f> DCM)
{
  DCM = q.toRotationMatrix();
}

inline __device__ void omega2edot(const float p, const float q, const float r, const float e[4], float ed[4])
{
  ed[0] = 0.5 * (-p * e[1] - q * e[2] - r * e[3]);
  ed[1] = 0.5 * (p * e[0] - q * e[3] + r * e[2]);
  ed[2] = 0.5 * (p * e[3] + q * e[0] - r * e[1]);
  ed[3] = 0.5 * (-p * e[2] + q * e[1] + r * e[0]);
}

// Can't use Eigen::Ref on Quaternions
inline void omega2edot(const float p, const float q, const float r, const Eigen::Quaternionf& e, Eigen::Quaternionf& ed)
{
  ed.w() = 0.5 * (-p * e.x() - q * e.y() - r * e.z());
  ed.x() = 0.5 * (p * e.w() - q * e.z() + r * e.y());
  ed.y() = 0.5 * (p * e.z() + q * e.w() - r * e.x());
  ed.z() = 0.5 * (-p * e.y() + q * e.x() + r * e.w());
}

inline __device__ __host__ Eigen::Matrix3f skewSymmetricMatrix(Eigen::Vector3f& v)
{
  Eigen::Matrix3f m;
  m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return m;
}

inline __host__ double timeDiffms(const std::chrono::steady_clock::time_point& end,
                                  const std::chrono::steady_clock::time_point& start)
{
  return (end - start).count() / 1e6;
}

inline __host__ __device__ double normalCDF(double x)
{
  return 0.5 * erfc(-x * M_SQRT1_2);
}

inline __host__ std::vector<double> calculateCk(size_t steps)
{
  // calculate params only when more steps are required
  static std::vector<double> c_vec;
  if (c_vec.size() < steps)
  {
    c_vec.resize(steps, 0);
    c_vec[0] = 1.0;
    for (size_t k = 1; k <= steps; k++)
    {
      double c_k = 0;
      for (size_t m = 0; m < k; m++)
      {
        c_k += c_vec[m] * c_vec[k - 1 - m] / ((m + 1.0) * (2.0 * m + 1.0));
      }
      c_vec[k] = c_k;
    }
  }
  return c_vec;
}

/**
 * Implementation based on
 * https://en.wikipedia.org/wiki/Error_function#Inverse_functions and Horner's
 * method
 */
inline __host__ double inverseErrorFunc(double x, int num_precision = 5)
{
  std::vector<double> c_k = calculateCk(num_precision);
  double output = 0;
  for (int i = num_precision; i > 0; i--)
  {
    output = (c_k[i] / (2.0 * i + 1.0) + output) * x * x * M_PI / 4.0;
  }
  output = (output + c_k[0]) * x / M_2_SQRTPI;
  return output;
}

inline __host__ double inverseErrorFuncSlow(double x, int num_precision = 5)
{
  std::vector<double> c_k = calculateCk(num_precision);
  double slow_output = 0;
  for (int i = 0; i <= num_precision; i++)
  {
    slow_output += c_k[i] / (2.0 * i + 1.0) * std::pow(x / M_2_SQRTPI, 2 * i + 1);
  }
  return slow_output;
}

/**
 * https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function
 */
inline __host__ double inverseNormalCDF(double x, int num_precision = 10)
{
  return M_SQRT2 * inverseErrorFunc(2.0 * x - 1.0, num_precision);
}

inline __host__ double inverseNormalCDFSlow(double x, int num_precision = 10)
{
  return M_SQRT2 * inverseErrorFuncSlow(2.0 * x - 1.0, num_precision);
}

}  // namespace math

// Matching float4 syntax
template <class T = float>
struct __align__(4 * sizeof(T)) type4
{
  T x;
  T y;
  T z;
  T w;
  // Allow writing to struct using array index
  __host__ __device__ T& operator[](int i)
  {
    assert(i >= 0);
    assert(i < 4);
    return (i > 1) ? ((i == 2) ? z : w) : ((i == 0) ? x : y);
  }
  // Allow reading from struct using array index
  __host__ __device__ const T& operator[](int i) const
  {
    assert(i >= 0);
    assert(i < 4);
    return (i > 1) ? ((i == 2) ? z : w) : ((i == 0) ? x : y);
  }
};

template <class T = float>
struct __align__(2 * sizeof(T)) type2
{
  T x;
  T y;

  // Allow writing to struct using array index
  __host__ __device__ T& operator[](int i)
  {
    assert(i >= 0);
    assert(i < 2);
    return (i == 0) ? x : y;
  }
  // Allow reading from struct using array index
  __host__ __device__ const T& operator[](int i) const
  {
    assert(i >= 0);
    assert(i < 2);
    return (i == 0) ? x : y;
  }
};

namespace p1  // parallelize using 1 thread dim
{
enum class Parallel1Dir : int
{
  THREAD_X = 0,
  THREAD_Y,
  THREAD_Z,
  GLOBAL_X,
  GLOBAL_Y,
  GLOBAL_Z,
  NONE,
};

template <Parallel1Dir P_DIR>
inline __host__ __device__ void getParallel1DIndex(int& p_index, int& p_step);

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_X>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.x;
  p_step = blockDim.x;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_Y>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.y;
  p_step = blockDim.y;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_Z>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.z;
  p_step = blockDim.z;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::GLOBAL_X>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.x + blockDim.x * blockIdx.x;
  p_step = gridDim.x * blockDim.x;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::GLOBAL_Y>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.y + blockDim.y * blockIdx.y;
  p_step = gridDim.y * blockDim.y;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::GLOBAL_Z>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.z + blockDim.z * blockIdx.z;
  p_step = gridDim.z * blockDim.z;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::NONE>(int& p_index, int& p_step)
{
  p_index = 0;
  p_step = 1;
}

template <int N, Parallel1Dir P_DIR = Parallel1Dir::THREAD_Y, class T = float>
inline __device__ void loadArrayParallel(T* __restrict__ a1, const int off1, const T* __restrict__ a2, const int off2)
{
  int p_index, p_step;
  getParallel1DIndex<P_DIR>(p_index, p_step);
  if (N % 4 == 0 && sizeof(type4<T>) < 16 && off1 % 4 == 0 && off2 % 4 == 0)
  {
    for (int i = p_index; i < N / 4; i += p_step)
    {
      reinterpret_cast<type4<T>*>(&a1[off1])[i] = reinterpret_cast<const type4<T>*>(&a2[off2])[i];
    }
  }
  else if (N % 2 == 0 && sizeof(type2<T>) < 16 && off1 % 2 == 0 && off2 % 2 == 0)
  {
    for (int i = p_index; i < N / 2; i += p_step)
    {
      reinterpret_cast<type2<T>*>(&a1[off1])[i] = reinterpret_cast<const type2<T>*>(&a2[off2])[i];
    }
  }
  else
  {
    for (int i = p_index; i < N; i += p_step)
    {
      a1[off1 + i] = a2[off2 + i];
    }
  }
}
}  // namespace p1

namespace p2  // parallelize using 2 thread dim
{
enum class Parallel2Dir : int
{
  THREAD_XY = 0,
  THREAD_XZ,
  THREAD_YZ,
  THREAD_YX,
  THREAD_ZX,
  THREAD_ZY,
  NONE
};

template <Parallel2Dir P_DIR>
inline __host__ __device__ void getParallel2DIndex(int& p1_index, int& p2_index, int& p1_step, int& p2_step)
{
#ifndef __CUDA_ARCH__
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_XY>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.x;
  p1_step = blockDim.x;
  p2_index = threadIdx.y;
  p2_step = blockDim.y;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_YZ>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.y;
  p1_step = blockDim.y;
  p2_index = threadIdx.z;
  p2_step = blockDim.z;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_XZ>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.x;
  p1_step = blockDim.x;
  p2_index = threadIdx.z;
  p2_step = blockDim.z;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_YX>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.y;
  p1_step = blockDim.y;
  p2_index = threadIdx.x;
  p2_step = blockDim.x;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_ZY>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.z;
  p1_step = blockDim.z;
  p2_index = threadIdx.y;
  p2_step = blockDim.y;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_ZX>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.z;
  p1_step = blockDim.z;
  p2_index = threadIdx.x;
  p2_step = blockDim.x;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::NONE>(int& p1_index, int& p2_index, int& p1_step, int& p2_step)
{
  p1_index = 0;
  p1_step = 1;
  p2_index = 0;
  p2_step = 1;
}
}  // namespace p2

namespace matrix_multiplication
{
/**
 * Utility Functions
 **/
inline __host__ __device__ int2 const unravelColumnMajor(const int index, const int num_rows)
{
  int col = index / num_rows;
  int row = index % num_rows;
  return make_int2(row, col);
}

inline __host__ __device__ int2 const unravelRowMajor(const int index, const int num_cols)
{
  int row = index / num_cols;
  int col = index % num_cols;
  return make_int2(row, col);
}
inline __host__ __device__ constexpr int columnMajorIndex(const int row, const int col, const int num_rows)
{
  return col * num_rows + row;
}

inline __host__ __device__ constexpr int rowMajorIndex(const int row, const int col, const int num_cols)
{
  return row * num_cols + col;
}

/**
 * Utility Classes
 **/
enum class MAT_OP : int
{
  NONE = 0,
  TRANSPOSE
};

template <int M, int N, class T = float>
class devMatrix
{
public:
  T* data = nullptr;
  static constexpr int rows = M;
  static constexpr int cols = N;
  devMatrix(T* n_data)
  {
    data = n_data;
  };

  T operator()(const int i, const int j) const
  {
    return data[columnMajorIndex(i, j, rows)];
  }
};

/**
 * @brief GEneral Matrix Multiplication
 * Conducts the operation
 * C = alpha * op(A) * op(B) + beta * C
 * on matrices of type T
 * TODO: Add transpose options like cuBLAS GEMM
 * Inputs:
 * op(A) - T-type column-major matrix of size M * K, stored in shared/global mem
 * op(B) - T-type column-major matrix of size K * N, stored in shared/global mem
 * alpha - T-type to multiply A * B
 * beta - T-type multipling C
 * A_OP - whether or not you should use A or A transpose
 * B_OP - whether or not you should use B or B transpose
 * Outputs:
 * C - float column-major matrix of size M * N, stored in shared/global mem
 *
 */
template <int M, int K, int N, p1::Parallel1Dir P_DIR = p1::Parallel1Dir::THREAD_Y, class T = float>
inline __device__ __host__ void gemm1(const T* A, const T* B, T* C, const T alpha = 1, const T beta = 0,
                                      const MAT_OP A_OP = MAT_OP::NONE, const MAT_OP B_OP = MAT_OP::NONE)
{
  int parallel_index;
  int parallel_step;
  int p, k;
  p1::getParallel1DIndex<P_DIR>(parallel_index, parallel_step);
  int2 mn;
  const bool all_stride = (A_OP == MAT_OP::NONE) && (B_OP == MAT_OP::TRANSPOSE);
  for (p = parallel_index; p < M * N; p += parallel_step)
  {
    T accumulator = 0;
    mn = unravelColumnMajor(p, M);
    if (K % 4 == 0 && sizeof(type4<T>) <= 16 && !all_stride)
    {  // Fetch 4 B values using single load memory operator of up to 128 bits since B is contiguous wrt k
      __UNROLL(10)
      for (k = 0; k < K; k += 4)
      {
        if (A_OP == MAT_OP::NONE && B_OP == MAT_OP::NONE)
        {
          const type4<T> b_tmp = reinterpret_cast<const type4<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          accumulator += A[columnMajorIndex(mn.x, k + 0, M)] * b_tmp[0];
          accumulator += A[columnMajorIndex(mn.x, k + 1, M)] * b_tmp[1];
          accumulator += A[columnMajorIndex(mn.x, k + 2, M)] * b_tmp[2];
          accumulator += A[columnMajorIndex(mn.x, k + 3, M)] * b_tmp[3];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::NONE)
        {
          const type4<T> b_tmp = reinterpret_cast<const type4<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          const type4<T> a_tmp = reinterpret_cast<const type4<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * b_tmp[0];
          accumulator += a_tmp[1] * b_tmp[1];
          accumulator += a_tmp[2] * b_tmp[2];
          accumulator += a_tmp[3] * b_tmp[3];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::TRANSPOSE)
        {
          // const type4<T> b_tmp = reinterpret_cast<const type4<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          const type4<T> a_tmp = reinterpret_cast<const type4<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * B[rowMajorIndex(k + 0, mn.y, N)];
          accumulator += a_tmp[1] * B[rowMajorIndex(k + 1, mn.y, N)];
          accumulator += a_tmp[2] * B[rowMajorIndex(k + 2, mn.y, N)];
          accumulator += a_tmp[3] * B[rowMajorIndex(k + 3, mn.y, N)];
        }
      }
    }
    else if (K % 2 == 0 && sizeof(type2<T>) <= 16 && !all_stride)
    {  // Fetch 2 B values using single load memory operator of up to 128 bits since B is contiguous wrt k
      __UNROLL(10)
      for (k = 0; k < K; k += 2)
      {
        if (A_OP == MAT_OP::NONE && B_OP == MAT_OP::NONE)
        {
          const type2<T> b_tmp = reinterpret_cast<const type2<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          accumulator += A[columnMajorIndex(mn.x, k + 0, M)] * b_tmp[0];
          accumulator += A[columnMajorIndex(mn.x, k + 1, M)] * b_tmp[1];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::NONE)
        {
          const type2<T> b_tmp = reinterpret_cast<const type2<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          const type2<T> a_tmp = reinterpret_cast<const type2<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * b_tmp[0];
          accumulator += a_tmp[1] * b_tmp[1];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::TRANSPOSE)
        {
          const type2<T> a_tmp = reinterpret_cast<const type2<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * B[rowMajorIndex(k + 0, mn.y, N)];
          accumulator += a_tmp[1] * B[rowMajorIndex(k + 1, mn.y, N)];
        }
      }
    }
    else
    {  // Either K is odd or sizeof(T) is large enough that
      T a;
      T b;
      __UNROLL(10)
      for (k = 0; k < K; k++)
      {
        if (A_OP == MAT_OP::NONE && B_OP == MAT_OP::NONE)
        {
          a = A[columnMajorIndex(mn.x, k, M)];
          b = B[columnMajorIndex(k, mn.y, K)];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::NONE)
        {
          a = A[rowMajorIndex(mn.x, k, K)];
          b = B[columnMajorIndex(k, mn.y, K)];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::TRANSPOSE)
        {
          a = A[rowMajorIndex(mn.x, k, K)];
          b = B[rowMajorIndex(k, mn.y, N)];
        }
        else
        {
          a = A[columnMajorIndex(mn.x, k, M)];
          b = B[rowMajorIndex(k, mn.y, N)];
        }

        accumulator += a * b;
      }
    }
    if (beta == 0)
    {  // Special case to remove extraneous memory accesses
      C[p] = alpha * accumulator;
    }
    else
    {
      C[p] = alpha * accumulator + beta * C[p];
    }
  }
}

/**
 * @brief GEneral Matrix Multiplication
 * Conducts the operation
 * C = alpha * A * B + beta * C
 * using two parallelization directions
 * TODO: Add transpose options like cuBLAS GEMM
 * Inputs:
 * A - float column-major matrix of size M * K, stored in shared/global mem
 * B - float column-major matrix of size K * N, stored in shared/global mem
 * alpha - float to multiply A * B
 * beta - float multipling C
 * Outputs:
 * C - float column-major matrix of size M * N, stored in shared/global mem
 *
 */
template <int M, int K, int N, p2::Parallel2Dir P_DIR = p2::Parallel2Dir::THREAD_XY>
inline __device__ void gemm2(const float* A, const float* B, float* C, const float alpha = 1.0, const float beta = 0.0)
{
  int m_ind_start;
  int m_ind_size;
  int n_ind_start;
  int n_ind_size;
  p2::getParallel2DIndex<P_DIR>(m_ind_start, n_ind_start, m_ind_size, n_ind_size);
  for (int m = m_ind_start; m < M; m += m_ind_size)
  {
    for (int n = n_ind_start; n < N; n += n_ind_size)
    {
      float accumulator = 0;
      __UNROLL(10)
      for (int k = 0; k < K; k++)
      {
        accumulator += A[columnMajorIndex(m, k, M)] * B[columnMajorIndex(k, n, K)];
      }
      C[columnMajorIndex(m, n, M)] = alpha * accumulator + beta * C[columnMajorIndex(m, n, M)];
    }
  }
}

template <p1::Parallel1Dir P_DIR = p1::Parallel1Dir::NONE, int M = 1, int K = 1, int N = 1, class T = float>
void matMult1(const devMatrix<M, K, T>& A, const devMatrix<K, N, T>& B, devMatrix<M, N, T>& C)
{
  gemm1<M, K, N, P_DIR, T>(A.data, B.data, C.data);
}

}  // namespace matrix_multiplication
}  // namespace mppi

#endif  // MATH_UTILS_H_
