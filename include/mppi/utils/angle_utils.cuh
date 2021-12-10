#ifndef ANGLES_CUH_
#define ANGLES_CUH_

namespace angle_utils
{
/**
 *
 * @param angle
 * @return
 */
__host__ __device__ static inline double normalizeAngle(double angle)
{
  const double result = fmod(angle + M_PI, 2.0 * M_PI);
  if (result <= 0.0)
    return result + M_PI;
  return result - M_PI;
}
__host__ __device__ static inline float normalizeAngle(float angle)
{
  const float result = fmodf(angle + M_PI, 2.0 * M_PI);
  if (result <= 0.0)
    return result + M_PI;
  return result - M_PI;
}

/**
 *
 * @param from
 * @param to
 * @return
 */
__host__ __device__ static inline double shortestAngularDistance(double from, double to)
{
  return normalizeAngle(to - from);
}
__host__ __device__ static inline float shortestAngularDistance(float from, float to)
{
  return normalizeAngle(to - from);
}

/**
 * Does a racer_dubins interpolation of the euler angle while respecting -pi to pi wrapping
 * solution from https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf algorithm 6
 * @param angle_1
 * @param angle_2
 * @param alpha
 * @return
 */
__host__ __device__ static inline double interpolateEulerAngleLinear(double angle_1, double angle_2, double alpha)
{
  double angle_diff = shortestAngularDistance(angle_1, angle_2);
  return normalizeAngle(angle_1 + alpha * angle_diff);
}
__host__ __device__ static inline float interpolateEulerAngleLinear(float angle_1, float angle_2, float alpha)
{
  float angle_diff = shortestAngularDistance(angle_1, angle_2);
  return normalizeAngle(angle_1 + alpha * angle_diff);
}
}  // namespace angle_utils

#endif  // ANGLES_CUH_
