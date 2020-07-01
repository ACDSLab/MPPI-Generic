/*
 * Created on Tue Jun 30 2020 by Bogdan
 */
#ifndef FEEDBACK_CONTROLLERS_CCM_CCM_H_
#define FEEDBACK_CONTROLLERS_CCM_CCM_H_

#include <Eigen/Dense>

#include <tuple>
#include <cmath>

namespace ccm {
// Convienient Types
template<int N = 1>
using Vectorf = Eigen::Matrix<float, N, 1>;

// ================ Chebyshev Methods =================
// Can also be written as pts, and weights being passed in.
template <int N = 1>
std::tuple<Vectorf<N>,Vectorf<N>> chebyshevPts() {
  Vectorf<N> pts, weights;
  int K = N - 1;
  for (int i = K; i >= 0; i--) {
    pts[i] = (cosf(M_PI * i / K) + 1) / 2.0;
  }
  // Weights calculations. Weights look to be symmetric
  int Kh = K * K - (K % 2 == 0) * 1;
  weights[0] = 0.5 / Kh;
  weights[K] = 0.5 / Kh;
  int K_iteration = K / 2;
  for (int k = 1; k <= K_iteration; k++) {
    float w_k = 0;
    for (int j = 0; j <= K_iteration; j++) {
      int beta = 2;
      if (j == 0 || j == K_iteration) {
        beta = 1;
      }
      w_k += beta * cosf((M_2_PI * j * k) / K) / (K * (1 - 4 * j * j));
    }
    weights[k] = w_k;
    weights[K - k] = w_k;
  }
  return std::make_tuple(pts, weights);
}

// Create a polynomial matrix given a set of points and the max degree of the polynomials
template<int N = 1, int D = 2>
Eigen::Matrix<float, D, N> chebyshevPolynomial(const Eigen::Ref<const Vectorf<N>>& pts) {
  Eigen::Matrix<float, D, N> T = Eigen::Matrix<float, D, N>::Zero();
  T.row(0) = Vectorf<N>::Ones();
  T.row(1) = pts;
  for (int i = 2; i < D; i++) {
    for (int j = 0; j < N; j++) {
      T(i, j) = 2 * pts(j) * T(i - 1, j) - T(i - 2, j);
    }
  }
  return T;
}

// Create the derivative of the polynomial matrix provided
template<int N = 1, int D = 2>
Eigen::Matrix<float, D, N> chebyshevPolynomialDerivative(
    const Eigen::Ref<const Eigen::Matrix<float, D, N>>& T,
    const Eigen::Ref<const Vectorf<N>>& pts) {
  Eigen::Matrix<float, D, N> dT = Eigen::Matrix<float, D, N>::Zero();
  dT.row(1) = Vectorf<N>::Ones();
  for (int i = 2; i < D; i++) {
    for (int j = 0; j < N; j++) {
      dT(i, j) = 2 * T(i - 1, j) + 2 * pts(j) * dT(i - 1, j) - dT(i - 2, j);
    }
  }
  return dT;
}


} // namespace ccm

#endif  // FEEDBACK_CONTROLLERS_CCM_CCM_H_