#include <array>

#include <gtest/gtest.h>
#include <mppi/utils/test_helper.h>
#include <mppi/utils/math_utils.h>

TEST(MATH_UTILS, QuatInv)
{
  // Create an unnormalized quaternion.
  std::array<float, 4> q{ 1, 2, 3, 2 };
  std::array<float, 4> q_inv{};

  mppi_math::QuatInv(q.data(), q_inv.data());

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

  mppi_math::QuatMultiply(q1.data(), q2.data(), q3.data());

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
  eigen_assert_float_eq<Eigen::Vector3f>(a.cross(b), mppi_math::skewSymmetricMatrix(a) * b);
}
