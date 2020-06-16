/*
 * Created on Mon Jun 01 2020 by Bogdan
 *
 */
#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

// Needed for sampling without replacement
#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <Eigen/Dense>

#ifndef SQ
#define SQ(a) a * a
#endif // SQ

namespace mppi_math {
// Based off of https://gormanalysis.com/blog/random-numbers-in-cpp
inline std::vector<int> sample_without_replacement(int k, int N,
    std::default_random_engine g = std::default_random_engine()) {
  if (k > N) {
    throw std::logic_error("Can't sample more than n times without replacement");
  }
  // Create an unordered set to store the samples
  std::unordered_set<int> samples;

  // For loop runs k times
  for (int r = N - k; r < N; r++) {
    int v = std::uniform_int_distribution<>(1, r)(g); // sample between 1 and r
    if (!samples.insert(v).second) { // if v exists in the set
      samples.insert(r);
    }
  }
  // Copy set into a vector
  std::vector<int> final_sequence(samples.begin(), samples.end());
  // Shuffle the vector to get the final sequence of sampling
  std::shuffle(final_sequence.begin(), final_sequence.end(), g);
  return final_sequence;
}

inline __device__ void Quat2DCM(float q[4], float M[3][3]) {
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

inline void Quat2DCM(const Eigen::Quaternionf& q, Eigen::Ref<Eigen::Matrix3f> DCM) {
  DCM = q.toRotationMatrix();
}

inline __device__ void omega2edot(const float p, const float q, const float r,
                                  const float e[4], float ed[4]) {
  ed[0] = 0.5 * (-p * e[1] - q * e[2] - r * e[3]);
  ed[1] = 0.5 * ( p * e[0] - q * e[3] + r * e[2]);
  ed[2] = 0.5 * ( p * e[3] + q * e[0] - r * e[1]);
  ed[3] = 0.5 * (-p * e[2] + q * e[1] + r * e[0]);
}

// Can't use Eigen::Ref on Quaternions
inline void omega2edot(const float p, const float q, const float r,
                       const Eigen::Quaternionf& e,
                       Eigen::Quaternionf& ed) {
  ed.w() = 0.5 * (-p * e.x() - q * e.y() - r * e.z());
  ed.x() = 0.5 * ( p * e.w() - q * e.z() + r * e.y());
  ed.y() = 0.5 * ( p * e.z() + q * e.w() - r * e.x());
  ed.z() = 0.5 * (-p * e.y() + q * e.x() + r * e.w());
}
} // mppi_math

#endif // MATH_UTILS_H_
