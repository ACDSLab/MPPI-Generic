#ifndef ACTIVATION_FUNCTIONS_CUH_
#define ACTIVATION_FUNCTIONS_CUH_

namespace act_func
{
/**
 *
 * @param input
 * @return
 */
__host__ __device__ static inline double tanh_deriv(double x)
{
  const double tanh_res = tanh(x);
  return (1 - tanh_res * tanh_res);
}

/**
 *
 * @param x
 * @return
 */
__host__ __device__ static inline float tanh_deriv(float x)
{
  const float tanh_res = tanhf(x);
  return (1 - tanh_res * tanh_res);
}

/**
 *
 * @param x
 * @return
 */
__host__ __device__ static inline float tanhshrink(float x)
{
  return x - tanh(x);
}

/**
 *
 * @param x
 * @return
 */
__host__ __device__ static inline double tanhshrink(double x)
{
  return x - tanh(x);
}

/**
 *
 * @param x
 * @return
 */
__host__ __device__ static inline float tanhshrink_scale(float x, float scale)
{
  return scale * tanhshrink(x / scale);
}

/**
 *
 * @param x
 * @return
 */
__host__ __device__ static inline double tanhshrink_scale(double x, double scale)
{
  return scale * tanhshrink(x / scale);
}

/**
 *
 * @param x
 * @return
 */
__host__ __device__ static inline float relu(float x)
{
  return fmaxf(x, 0.0f);
}
}  // namespace act_func

#endif  // ACTIVATION_FUNCTIONS_CUH_
