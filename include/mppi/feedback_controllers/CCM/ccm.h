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

template<class DYN_T, int NUM_TIMESTEPS = 100>
class LinearCCM {
public:
  // Typedefs
  using state_array = typename DYN_T::state_array;
  using control_array = typename DYN_T::control_array;
  using B_matrix = typename DYN_T::dfdu;
  using RiemannianMetric = typename DYN_T::dfdx;

  typedef Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS> state_trajectory;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, NUM_TIMESTEPS> control_trajectory;

  LinearCCM(DYN_T* dyn) {
    model_ = dyn;
  }

  // Generic Method for calculating the Metric based on state x
  RiemannianMetric M(const Eigen::Ref<const state_array>& x) {
    return M_;
  }


  float Energy(const Eigen::Ref<const state_array>& delta_x,
               const Eigen::Ref<const state_array>& x) {
    return delta_x.transpose() * M(x) * delta_x;
  }

  // TODO Replace with some call to Dynamics
  B_matrix B(const Eigen::Ref<const state_array>& x) {
    return B_;
  }

  state_array f(const Eigen::Ref<const state_array>& x,
                const Eigen::Ref<const control_array>& u = control_array::Zero()) {
    state_array x_der;
    model_->computeStateDeriv(x, u, x_der);
    return x_der;
  }

  control_array u_feedback(state_array x_act, int t) {
    state_array x_nom_t = x_nominal_traj_.col(t);
    control_array u_nom_t = u_nominal_traj_.col(t);
    state_array delta_x = x_act - x_nom_t;

    float E = Energy(delta_x, x_act);
    control_array lhs = 2 * B(x_act).transpose() * M(x_act) * delta_x;
    float normalize_lhs = lhs.norm() * lhs.norm();
    float rhs = -2 * lambda_ * E - 2 * delta_x.transpose() * M(x_act) *
                (f(x_act) - f(x_nom_t)) + (B(x_act) - B(x_mon_t)) * u_nom_t);
    if (rhs > 0 || normalize_lhs == 0) {
      return control_array::Zero();
    } else {
      return rhs / normalize_lhs * lhs;
    }
  }

  void setNominalControlTrajectory(const Eigen::Ref<const control_trajectory>& u_traj) {
    u_nominal_traj_ = u_traj;
  }

  void setNominalStateTrajectory(const Eigen::Ref<const state_trajectory>& x_traj) {
    x_nominal_traj_ = x_traj;
  }

protected:
  state_trajectory x_nominal_traj_;
  control_trajectory u_nominal_traj_;
  float lambda_ = 1.0;

  RiemannianMetric M_;

  DYN_T* model_;
  B_matrix B_ = B_matrix::Zero();


};

} // namespace ccm

#endif  // FEEDBACK_CONTROLLERS_CCM_CCM_H_