/*
 * Created on Wed Jun 03 2020 by Bogdan
 *
 */
#ifndef QUATERNION_MATH_CUH_
#define QUATERNION_MATH_CUH_

// Needed for sampling without replacement
#include <cuda_runtime.h>
#include <Eigen/Dense>

#ifndef SQ
#define SQ(a) a * a
#endif // SQ

namespace mppi_math {

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
// TODO Check that Quaternions are actually passed through correctly
inline void omega2edot(const float p, const float q, const float r,
                       const Eigen::Quaternionf& e,
                       Eigen::Quaternionf& ed) {
  ed.w() = 0.5 * (-p * e.x() - q * e.y() - r * e.z());
  ed.x() = 0.5 * ( p * e.w() - q * e.z() + r * e.y());
  ed.y() = 0.5 * ( p * e.z() + q * e.w() - r * e.x());
  ed.z() = 0.5 * (-p * e.y() + q * e.x() + r * e.w());
}

} // mppi_math

#endif // QUATERNION_MATH_CUH_
