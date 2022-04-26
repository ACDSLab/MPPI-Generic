#include <Eigen/Dense>

Eigen::Vector3f cudaToEigen(const float3& v)
{
  return Eigen::Vector3f(v.x, v.y, v.z);
}

float3 EigenToCuda(const Eigen::Vector3f& v)
{
  return make_float3(v[0], v[1], v[2]);
}
