/*
 * Created on Wed Jun 03 2020 by Bogdan
 *
 */
#ifndef QUATERNION_MATH_CUH_
#define QUATERNION_MATH_CUH_

// Needed for sampling without replacement
#include <cuda_runtime.h>
#include <Eigen/Dense>

namespace mppi_math {

inline __device__ void Quat2DCM(float q[4], float DCM[3][3]) {

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
                       Eigen::Quaternionf ed) {
  ed.w() = 0.5 * (-p * e.x() - q * e.y() - r * e.z());
  ed.x() = 0.5 * ( p * e.w() - q * e.z() + r * e.y());
  ed.y() = 0.5 * ( p * e.z() + q * e.w() - r * e.x());
  ed.z() = 0.5 * (-p * e.y() + q * e.x() + r * e.w());
}

} // mppi_math

#endif // QUATERNION_MATH_CUH_