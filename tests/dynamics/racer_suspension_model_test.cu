#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/racer_suspension/racer_suspension.cuh>
#include <mppi/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <cuda_runtime.h>

class RacerSuspensionTest : public ::testing::Test
{
public:
  cudaStream_t stream;

  void SetUp() override
  {
    CudaCheckError();
    HANDLE_ERROR(cudaStreamCreate(&stream));
  }

  void TearDown() override
  {
    CudaCheckError();
    HANDLE_ERROR(cudaStreamDestroy(stream));
  }
};

TEST_F(RacerSuspensionTest, Template)
{
  auto dynamics = RacerSuspension(stream);
  EXPECT_EQ(15, RacerSuspension::STATE_DIM);
  EXPECT_EQ(2, RacerSuspension::CONTROL_DIM);
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
}

TEST_F(RacerSuspensionTest, BindStream)
{
  auto dynamics = RacerSuspension(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);
}



TEST_F(RacerSuspensionTest, OmegaJacobian)
{
  auto dynamics = RacerSuspension(stream);
  RacerSuspension::state_array x = RacerSuspension::state_array::Zero();
  x[RacerSuspension::STATE_QW] = 1;
  x[RacerSuspension::STATE_OMEGAX] = 0.1;
  x[RacerSuspension::STATE_OMEGAY] = -0.03;
  x[RacerSuspension::STATE_OMEGAZ] = 0.02;
  x[RacerSuspension::STATE_VX] = 2;
  RacerSuspension::state_array x_dot0 = RacerSuspension::state_array::Zero();
  RacerSuspension::state_array x_dot1 = RacerSuspension::state_array::Zero();
  RacerSuspension::control_array u = RacerSuspension::control_array::Zero();
  Eigen::Matrix3f omega_jac;
  float delta = 0.001;
  dynamics.computeDynamics(x, u, x_dot0, &omega_jac);
  for (int i=0; i < 3; i++) {
    x[RacerSuspension::STATE_OMEGA+i] += delta;
    dynamics.computeDynamics(x, u, x_dot1);
    x[RacerSuspension::STATE_OMEGA+i] -= delta;

    float abs_tol = 2;
    EXPECT_NEAR(omega_jac.col(i)[0], (x_dot1[RacerSuspension::STATE_OMEGAX] - x_dot0[RacerSuspension::STATE_OMEGAX])/delta, abs_tol);
    EXPECT_NEAR(omega_jac.col(i)[1], (x_dot1[RacerSuspension::STATE_OMEGAY] - x_dot0[RacerSuspension::STATE_OMEGAY])/delta, abs_tol);
    EXPECT_NEAR(omega_jac.col(i)[2], (x_dot1[RacerSuspension::STATE_OMEGAZ] - x_dot0[RacerSuspension::STATE_OMEGAZ])/delta, abs_tol);
  }

}

/*
float c_t = 1.3;
float c_b = 2.5;
float c_v = 3.7;
float c_0 = 4.9;
float wheel_base = 0.3;
 */

// TEST_F(RacerSuspensionTest, ComputeDynamics)
// {
//   RacerSuspension dynamics = RacerSuspension();
//   RacerDubinsParams params = dynamics.getParams();
//   RacerSuspension::state_array x = RacerSuspension::state_array::Zero();
//   RacerSuspension::control_array u = RacerSuspension::control_array::Zero();
//
//   // computeDynamics should not touch the roll/pitch element
//   RacerSuspension::state_array next_x = RacerSuspension::state_array::Ones() * 0.153;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9);
//   EXPECT_FLOAT_EQ(next_x(1), 0);
//   EXPECT_FLOAT_EQ(next_x(2), 0);
//   EXPECT_FLOAT_EQ(next_x(3), 0);
//   EXPECT_FLOAT_EQ(next_x(4), 0);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << 1, M_PI_2, 0, 3, 0, 0.5, -0.5;
//   u << 1, 0;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 1.3 - 3.7);
//   EXPECT_FLOAT_EQ(next_x(1), 0);
//   EXPECT_NEAR(next_x(2), 0, 1e-7);
//   EXPECT_FLOAT_EQ(next_x(3), 1);
//   EXPECT_FLOAT_EQ(next_x(4), 0);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << 1, 0, 0, 3, 0, 0.5, -0.5;
//   u << -1, 0;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 2.5 - 3.7);
//   EXPECT_FLOAT_EQ(next_x(1), 0);
//   EXPECT_FLOAT_EQ(next_x(2), 1);
//   EXPECT_FLOAT_EQ(next_x(3), 0);
//   EXPECT_FLOAT_EQ(next_x(4), 0);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << -1, 0, 0, 3, 0, 0.5, -0.5;
//   u << 1, 0;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 3.7 + 1.3);
//   EXPECT_FLOAT_EQ(next_x(1), 0);
//   EXPECT_FLOAT_EQ(next_x(2), -1);
//   EXPECT_FLOAT_EQ(next_x(3), 0);
//   EXPECT_FLOAT_EQ(next_x(4), 0);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << -1, 0, 0, 3, 0, 0.5, -0.5;
//   u << -1, 0;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 2.5 + 3.7);
//   EXPECT_FLOAT_EQ(next_x(1), 0);
//   EXPECT_FLOAT_EQ(next_x(2), -1);
//   EXPECT_FLOAT_EQ(next_x(3), 0);
//   EXPECT_FLOAT_EQ(next_x(4), 0);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << -3, 0, 0, 3, 0, 0.5, -0.5;
//   u << -1, 0;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 + 2.5 * 3 + 3.7 * 3);
//   EXPECT_FLOAT_EQ(next_x(1), 0);
//   EXPECT_FLOAT_EQ(next_x(2), -3);
//   EXPECT_FLOAT_EQ(next_x(3), 0);
//   EXPECT_FLOAT_EQ(next_x(4), 0);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << 4, 0, 0, 3, 0, 0.5, -0.5;
//   u << -1, 0;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 2.5 * 4 - 3.7 * 4);
//   EXPECT_FLOAT_EQ(next_x(1), 0);
//   EXPECT_FLOAT_EQ(next_x(2), 4);
//   EXPECT_FLOAT_EQ(next_x(3), 0);
//   EXPECT_FLOAT_EQ(next_x(4), 0);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << 1, M_PI, 0, 3, 0, 0.5, -0.5;
//   u << 0, 1;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.7);
//   EXPECT_FLOAT_EQ(next_x(1), (1 / .3) * tan(0));
//   EXPECT_FLOAT_EQ(next_x(2), -1);
//   EXPECT_NEAR(next_x(3), 0, 1e-7);
//   EXPECT_FLOAT_EQ(next_x(4), -1 / 2.45);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
//
//   x << 1, M_PI, 0, 0, 0.5, 0.5, -0.5;
//   u << 1, -1;
//   dynamics.computeDynamics(x, u, next_x);
//   EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.7 + 1.3);
//   EXPECT_FLOAT_EQ(next_x(1), (1 / .3) * tan(0.5));
//   EXPECT_FLOAT_EQ(next_x(2), -1);
//   EXPECT_NEAR(next_x(3), 0, 1e-7);
//   EXPECT_FLOAT_EQ(next_x(4), 1 / 2.45);
//   EXPECT_FLOAT_EQ(next_x(5), 0.153);
//   EXPECT_FLOAT_EQ(next_x(6), 0.153);
// }
//
// TEST_F(RacerSuspensionTest, TestModelGPU)
// {
//   RacerSuspension dynamics = RacerSuspension();
//   dynamics.GPUSetup();
//   CudaCheckError();
//
//   Eigen::Matrix<float, RacerSuspension::CONTROL_DIM, 100> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, RacerSuspension::CONTROL_DIM, 100>::Random();
//   Eigen::Matrix<float, RacerSuspension::STATE_DIM, 100> state_trajectory;
//   state_trajectory = Eigen::Matrix<float, RacerSuspension::STATE_DIM, 100>::Random();
//
//   std::vector<std::array<float, 7>> s(100);
//   std::vector<std::array<float, 7>> s_der(100);
//   // steering, throttle
//   std::vector<std::array<float, 2>> u(100);
//   for (int state_index = 0; state_index < s.size(); state_index++)
//   {
//     for (int dim = 0; dim < s[0].size(); dim++)
//     {
//       s[state_index][dim] = state_trajectory.col(state_index)(dim);
//     }
//     for (int dim = 0; dim < u[0].size(); dim++)
//     {
//       u[state_index][dim] = control_trajectory.col(state_index)(dim);
//     }
//   }
//
//   // These variables will be changed so initialized to the right size only
//
//   // Run dynamics on dynamicsU
//   // Run dynamics on GPU
//   for (int y_dim = 1; y_dim <= 4; y_dim++)
//   {
//     launchComputeDynamicsTestKernel<RacerSuspension, 7, 2>(dynamics, s, u, s_der, y_dim);
//     for (int point = 0; point < 100; point++)
//     {
//       RacerSuspension::state_array state = state_trajectory.col(point);
//       RacerSuspension::control_array control = control_trajectory.col(point);
//       RacerSuspension::state_array state_der_cpu = RacerSuspension::state_array::Zero();
//
//       dynamics.computeDynamics(state, control, state_der_cpu);
//       for (int dim = 0; dim < RacerSuspension::STATE_DIM; dim++)
//       {
//         EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-5) << "at index " << point << " with y_dim " << y_dim;
//         EXPECT_TRUE(isfinite(s_der[point][dim]));
//       }
//     }
//   }
//
//   dynamics.freeCudaMem();
//   CudaCheckError();
// }
//
// TEST_F(RacerSuspensionTest, TestUpdateState)
// {
//   CudaCheckError();
//   RacerSuspension dynamics = RacerSuspension();
//   RacerSuspension::state_array state;
//   RacerSuspension::state_array state_der;
//
//   // TODO add in the elevation map
//
//   state << 0, 0, 0, 0, 0, -0.5, 0.5;
//   state_der << 1, 1, 1, 1, 1, 0, 0;
//   dynamics.updateState(state, state_der, 0.1);
//   EXPECT_TRUE(state_der == RacerSuspension::state_array::Zero());
//   EXPECT_FLOAT_EQ(state(0), 0.1);
//   EXPECT_FLOAT_EQ(state(1), 0.1);
//   EXPECT_FLOAT_EQ(state(2), 0.1);
//   EXPECT_FLOAT_EQ(state(3), 0.1);
//   EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 0.1));
//   EXPECT_FLOAT_EQ(state(5), -0.5);
//   EXPECT_FLOAT_EQ(state(6), 0.5);
//
//   state << 0, M_PI - 0.1, 0, 0, 0, -0.5, 0.5;
//   state_der << 1, 1, 1, 1, 1;
//   dynamics.updateState(state, state_der, 1.0);
//   EXPECT_TRUE(state_der == RacerSuspension::state_array::Zero());
//   EXPECT_FLOAT_EQ(state(0), 1.0);
//   EXPECT_FLOAT_EQ(state(1), 1.0 - M_PI - 0.1);
//   EXPECT_FLOAT_EQ(state(2), 1.0);
//   EXPECT_FLOAT_EQ(state(3), 1.0);
//   EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 1.0));
//   EXPECT_FLOAT_EQ(state(5), -0.5);
//   EXPECT_FLOAT_EQ(state(6), 0.5);
//
//   state << 0, -M_PI + 0.1, 0, 0, 0, -0.5, 0.5;
//   state_der << 1, -1, 1, 1, 1;
//   dynamics.updateState(state, state_der, 1.0);
//   EXPECT_TRUE(state_der == RacerSuspension::state_array::Zero());
//   EXPECT_FLOAT_EQ(state(0), 1.0);
//   EXPECT_FLOAT_EQ(state(1), M_PI + 0.1 - 1.0);
//   EXPECT_FLOAT_EQ(state(2), 1.0);
//   EXPECT_FLOAT_EQ(state(3), 1.0);
//   EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 1.0));
//   EXPECT_FLOAT_EQ(state(5), -0.5);
//   EXPECT_FLOAT_EQ(state(6), 0.5);
//
//   CudaCheckError();
// }
//
// TEST_F(RacerSuspensionTest, TestUpdateStateGPU)
// {
//   CudaCheckError();
//   RacerSuspension dynamics = RacerSuspension();
//   CudaCheckError();
//   dynamics.GPUSetup();
//   CudaCheckError();
//
//   Eigen::Matrix<float, RacerSuspension::CONTROL_DIM, 100> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, RacerSuspension::CONTROL_DIM, 100>::Random();
//   Eigen::Matrix<float, RacerSuspension::STATE_DIM, 100> state_trajectory;
//   state_trajectory = Eigen::Matrix<float, RacerSuspension::STATE_DIM, 100>::Random();
//
//   std::vector<std::array<float, 7>> s(100);
//   std::vector<std::array<float, 7>> s_der(100);
//   // steering, throttle
//   std::vector<std::array<float, 2>> u(100);
//
//   RacerSuspension::state_array state;
//   RacerSuspension::control_array control;
//   RacerSuspension::state_array state_der_cpu = RacerSuspension::state_array::Zero();
//
//   // Run dynamics on dynamicsU
//   // Run dynamics on GPU
//   for (int y_dim = 1; y_dim <= 10; y_dim++)
//   {
//     for (int state_index = 0; state_index < s.size(); state_index++)
//     {
//       for (int dim = 0; dim < s[0].size(); dim++)
//       {
//         s[state_index][dim] = state_trajectory.col(state_index)(dim);
//       }
//       for (int dim = 0; dim < u[0].size(); dim++)
//       {
//         u[state_index][dim] = control_trajectory.col(state_index)(dim);
//       }
//     }
//
//     launchComputeStateDerivTestKernel<RacerSuspension, RacerSuspension::STATE_DIM,
//                                       RacerSuspension::CONTROL_DIM>(dynamics, s, u, s_der, y_dim);
//     launchUpdateStateTestKernel<RacerSuspension, RacerSuspension::STATE_DIM>(dynamics, s, s_der, 0.1f, y_dim);
//     for (int point = 0; point < 100; point++)
//     {
//       RacerSuspension::state_array state = state_trajectory.col(point);
//       RacerSuspension::control_array control = control_trajectory.col(point);
//       RacerSuspension::state_array state_der_cpu = RacerSuspension::state_array::Zero();
//
//       dynamics.computeDynamics(state, control, state_der_cpu);
//       dynamics.updateState(state, state_der_cpu, 0.1f);
//       for (int dim = 0; dim < RacerSuspension::STATE_DIM; dim++)
//       {
//         if (dim < 5)
//         {
//           EXPECT_FLOAT_EQ(state_der_cpu(dim), s_der[point][dim]) << "at index " << point << " with y_dim " << y_dim;
//           EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
//           EXPECT_TRUE(isfinite(s[point][dim]));
//         }
//         else
//         {
//           EXPECT_FLOAT_EQ(state_der_cpu(dim), s_der[point][dim]) << "at index " << point << " with y_dim " << y_dim;
//           EXPECT_NEAR(s[point][dim], 0.0, 1e-4)
//               << "at index " << point << " with y_dim " << y_dim << " state index " << dim;
//           EXPECT_TRUE(isfinite(s[point][dim]));
//         }
//       }
//     }
//   }
//   dynamics.freeCudaMem();
// }
//
// TEST_F(RacerSuspensionTest, ComputeStateTrajectoryFiniteTest)
// {
//   RacerSuspension dynamics = RacerSuspension();
//   RacerDubinsParams params;
//   params.c_t = 3.0;
//   params.c_b = 0.2;
//   params.c_v = 0.2;
//   params.c_0 = 0.2;
//   params.wheel_base = 3.0;
//   params.steering_constant = 1.0;
//   dynamics.setParams(params);
//
//   Eigen::Matrix<float, RacerSuspension::CONTROL_DIM, 500> control_trajectory;
//   control_trajectory = Eigen::Matrix<float, RacerSuspension::CONTROL_DIM, 500>::Zero();
//   Eigen::Matrix<float, RacerSuspension::STATE_DIM, 500> state_trajectory;
//   state_trajectory = Eigen::Matrix<float, RacerSuspension::STATE_DIM, 500>::Zero();
//   RacerSuspension::state_array state_der;
//   RacerSuspension::state_array x;
//   x << 0, 1.46919e-6, 0.0140179, 1.09739e-8, -0.000735827;
//
//   for (int i = 0; i < 500; i++)
//   {
//     RacerSuspension::control_array u = control_trajectory.col(i);
//     dynamics.computeDynamics(x, u, state_der);
//     EXPECT_TRUE(x.allFinite());
//     EXPECT_TRUE(u.allFinite());
//     EXPECT_TRUE(state_der.allFinite());
//     dynamics.updateState(x, state_der, 0.02);
//     EXPECT_TRUE(x.allFinite());
//     EXPECT_TRUE(u.allFinite());
//     EXPECT_TRUE(state_der == RacerSuspension::state_array::Zero());
//   }
//   params.steering_constant = 0.5;
//   dynamics.setParams(params);
//
//   x << 0, 1.46919e-6, 0.0140179, 1.09739e-8, -1.0;
//   for (int i = 0; i < 500; i++)
//   {
//     RacerSuspension::control_array u = control_trajectory.col(i);
//     dynamics.computeDynamics(x, u, state_der);
//     EXPECT_TRUE(x.allFinite());
//     EXPECT_TRUE(u.allFinite());
//     EXPECT_TRUE(state_der.allFinite());
//     dynamics.updateState(x, state_der, 0.02);
//     EXPECT_TRUE(x.allFinite());
//     EXPECT_TRUE(u.allFinite());
//     EXPECT_TRUE(state_der == RacerSuspension::state_array::Zero());
//   }
// }
//
// /*
// class LinearDummy : public RacerSuspension {
// public:
//   bool computeGrad(const Eigen::Ref<const state_array> & state,
//                    const Eigen::Ref<const control_array>& control,
//                    Eigen::Ref<dfdx> A,
//                    Eigen::Ref<dfdu> B) {
//     return false;
//   };
// };
//
// TEST_F(RacerSuspensionTest, TestComputeGradComputation) {
//   Eigen::Matrix<float, RacerSuspension::STATE_DIM, RacerSuspension::STATE_DIM +
// RacerSuspension::CONTROL_DIM> numeric_jac; Eigen::Matrix<float, RacerSuspension::STATE_DIM,
// RacerSuspension::STATE_DIM + RacerSuspension::CONTROL_DIM> analytic_jac; RacerSuspension::state_array
// state; state << 1, 2, 3, 4; RacerSuspension::control_array control; control << 5;
//
//   auto analytic_grad_model = RacerSuspension();
//
//   RacerSuspension::dfdx A_analytic = RacerSuspension::dfdx::Zero();
//   RacerSuspension::dfdu B_analytic = RacerSuspension::dfdu::Zero();
//
//   analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);
//
//   auto numerical_grad_model = LinearDummy();
//
//   std::shared_ptr<ModelWrapperDDP<LinearDummy>> ddp_model =
// std::make_shared<ModelWrapperDDP<LinearDummy>>(&numerical_grad_model);
//
//   analytic_jac.leftCols<RacerSuspension::STATE_DIM>() = A_analytic;
//   analytic_jac.rightCols<RacerSuspension::CONTROL_DIM>() = B_analytic;
//   numeric_jac = ddp_model->df(state, control);
//
//   ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n" << numeric_jac << "\nAnalytic
//   Jacobian\n"
// << analytic_jac;
// }
//
// */
