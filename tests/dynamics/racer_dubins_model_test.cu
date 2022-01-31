#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <cuda_runtime.h>

TEST(RacerDubins, Template)
{
  auto dynamics = RacerDubins();
  EXPECT_EQ(5, RacerDubins::STATE_DIM);
  EXPECT_EQ(2, RacerDubins::CONTROL_DIM);
}

TEST(RacerDubins, BindStream)
{
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  auto dynamics = RacerDubins(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

/*
float c_t = 1.3;
float c_b = 2.5;
float c_v = 3.7;
float c_0 = 4.9;
float wheel_base = 0.3;
 */

TEST(RacerDubins, ComputeDynamics)
{
  RacerDubins dynamics = RacerDubins();
  RacerDubinsParams params = dynamics.getParams();
  RacerDubins::state_array x = RacerDubins::state_array::Zero();
  RacerDubins::control_array u = RacerDubins::control_array::Zero();

  RacerDubins::state_array next_x;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), 0);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);

  x << 1, M_PI_2, 0, 3, 0;
  u << 1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 + 1.3 - 3.7);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_NEAR(next_x(2), 0, 1e-7);
  EXPECT_FLOAT_EQ(next_x(3), 1);
  EXPECT_FLOAT_EQ(next_x(4), 0);

  x << 1, 0, 0, 3, 0;
  u << -1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 - 2.5 - 3.7);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), 1);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);

  x << -1, 0, 0, 3, 0;
  u << 1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 + 3.7 - 1.3);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);

  x << -1, 0, 0, 3, 0;
  u << -1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 + 2.5 + 3.7);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);

  x << 1, M_PI, 0, 3, 0;
  u << 0, 1;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.7);
  EXPECT_FLOAT_EQ(next_x(1), (1 / .3) * tan(0));
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_NEAR(next_x(3), 0, 1e-7);
  EXPECT_FLOAT_EQ(next_x(4), 1 * 0.4);

  x << 1, M_PI, 0, 0, 0.5;
  u << 1, -1;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.7 + 1.3);
  EXPECT_FLOAT_EQ(next_x(1), (1 / .3) * tan(0.5));
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_NEAR(next_x(3), 0, 1e-7);
  EXPECT_FLOAT_EQ(next_x(4), -1 * 0.4);
}

TEST(RacerDubins, TestModelGPU)
{
  RacerDubins dynamics = RacerDubins();
  dynamics.GPUSetup();

  Eigen::Matrix<float, RacerDubins::CONTROL_DIM, 100> control_trajectory;
  control_trajectory = Eigen::Matrix<float, RacerDubins::CONTROL_DIM, 100>::Random();
  Eigen::Matrix<float, RacerDubins::STATE_DIM, 100> state_trajectory;
  state_trajectory = Eigen::Matrix<float, RacerDubins::STATE_DIM, 100>::Random();

  std::vector<std::array<float, 5>> s(100);
  std::vector<std::array<float, 5>> s_der(100);
  // steering, throttle
  std::vector<std::array<float, 3>> u(100);
  for (int state_index = 0; state_index < s.size(); state_index++)
  {
    for (int dim = 0; dim < s[0].size(); dim++)
    {
      s[state_index][dim] = state_trajectory.col(state_index)(dim);
    }
    for (int dim = 0; dim < u[0].size(); dim++)
    {
      u[state_index][dim] = control_trajectory.col(state_index)(dim);
    }
  }

  // These variables will be changed so initialized to the right size only

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for (int y_dim = 1; y_dim <= 4; y_dim++)
  {
    launchComputeDynamicsTestKernel<RacerDubins, 5, 3>(dynamics, s, u, s_der, y_dim);
    for (int point = 0; point < 100; point++)
    {
      RacerDubins::state_array state = state_trajectory.col(point);
      RacerDubins::control_array control = control_trajectory.col(point);
      RacerDubins::state_array state_der_cpu = RacerDubins::state_array::Zero();

      dynamics.computeDynamics(state, control, state_der_cpu);
      for (int dim = 0; dim < RacerDubins::STATE_DIM; dim++)
      {
        EXPECT_FLOAT_EQ(state_der_cpu(dim), s_der[point][dim]) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_TRUE(isfinite(s_der[point][dim]));
      }
    }
  }

  dynamics.freeCudaMem();
}

TEST(RacerDubins, TestUpdateState)
{
  RacerDubins dynamics = RacerDubins();
  RacerDubins::state_array state;
  RacerDubins::state_array state_der;

  state << 0, 0, 0, 0, 0;
  state_der << 1, 1, 1, 1, 1;
  dynamics.updateState(state, state_der, 0.1);
  EXPECT_TRUE(state_der == RacerDubins::state_array::Zero());
  EXPECT_FLOAT_EQ(state(0), 0.1);
  EXPECT_FLOAT_EQ(state(1), 0.1);
  EXPECT_FLOAT_EQ(state(2), 0.1);
  EXPECT_FLOAT_EQ(state(3), 0.1);
  EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 0.1));

  state << 0, M_PI - 0.1, 0, 0, 0;
  state_der << 1, 1, 1, 1, 1;
  dynamics.updateState(state, state_der, 1.0);
  EXPECT_TRUE(state_der == RacerDubins::state_array::Zero());
  EXPECT_FLOAT_EQ(state(0), 1.0);
  EXPECT_FLOAT_EQ(state(1), 1.0 - M_PI - 0.1);
  EXPECT_FLOAT_EQ(state(2), 1.0);
  EXPECT_FLOAT_EQ(state(3), 1.0);
  EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 1.0));

  state << 0, -M_PI + 0.1, 0, 0, 0;
  state_der << 1, -1, 1, 1, 1;
  dynamics.updateState(state, state_der, 1.0);
  EXPECT_TRUE(state_der == RacerDubins::state_array::Zero());
  EXPECT_FLOAT_EQ(state(0), 1.0);
  EXPECT_FLOAT_EQ(state(1), M_PI + 0.1 - 1.0);
  EXPECT_FLOAT_EQ(state(2), 1.0);
  EXPECT_FLOAT_EQ(state(3), 1.0);
  EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 1.0));
}

TEST(RacerDubins, TestUpdateStateGPU)
{
  RacerDubins dynamics = RacerDubins();
  dynamics.GPUSetup();

  Eigen::Matrix<float, RacerDubins::CONTROL_DIM, 100> control_trajectory;
  control_trajectory = Eigen::Matrix<float, RacerDubins::CONTROL_DIM, 100>::Random();
  Eigen::Matrix<float, RacerDubins::STATE_DIM, 100> state_trajectory;
  state_trajectory = Eigen::Matrix<float, RacerDubins::STATE_DIM, 100>::Random();

  std::vector<std::array<float, 5>> s(100);
  std::vector<std::array<float, 5>> s_der(100);
  // steering, throttle
  std::vector<std::array<float, 2>> u(100);

  RacerDubins::state_array state;
  RacerDubins::control_array control;
  RacerDubins::state_array state_der_cpu = RacerDubins::state_array::Zero();

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for (int y_dim = 1; y_dim <= 4; y_dim++)
  {
    for (int state_index = 0; state_index < s.size(); state_index++)
    {
      for (int dim = 0; dim < s[0].size(); dim++)
      {
        s[state_index][dim] = state_trajectory.col(state_index)(dim);
      }
      for (int dim = 0; dim < u[0].size(); dim++)
      {
        u[state_index][dim] = control_trajectory.col(state_index)(dim);
      }
    }

    launchComputeStateDerivTestKernel<RacerDubins, RacerDubins::STATE_DIM, RacerDubins::CONTROL_DIM>(dynamics, s, u,
                                                                                                     s_der, y_dim);
    launchUpdateStateTestKernel<RacerDubins, RacerDubins::STATE_DIM>(dynamics, s, s_der, 0.1f, y_dim);
    for (int point = 0; point < 100; point++)
    {
      RacerDubins::state_array state = state_trajectory.col(point);
      RacerDubins::control_array control = control_trajectory.col(point);
      RacerDubins::state_array state_der_cpu = RacerDubins::state_array::Zero();

      dynamics.computeDynamics(state, control, state_der_cpu);
      dynamics.updateState(state, state_der_cpu, 0.1f);
      for (int dim = 0; dim < RacerDubins::STATE_DIM; dim++)
      {
        EXPECT_FLOAT_EQ(state_der_cpu(dim), s_der[point][dim]) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_TRUE(isfinite(s[point][dim]));
      }
    }
  }
  dynamics.freeCudaMem();
}

TEST(RacerDubins, ComputeStateTrajectoryTest)
{
  RacerDubins dynamics = RacerDubins();
  RacerDubinsParams params;
  params.c_t = 3.0;
  params.c_b = 0.2;
  params.c_v = 0.2;
  params.c_0 = 0.2;
  params.wheel_base = 3.0;
  params.steering_constant = 1.0;
  dynamics.setParams(params);

  Eigen::Matrix<float, RacerDubins::CONTROL_DIM, 500> control_trajectory;
  control_trajectory = Eigen::Matrix<float, RacerDubins::CONTROL_DIM, 500>::Zero();
  Eigen::Matrix<float, RacerDubins::STATE_DIM, 500> state_trajectory;
  state_trajectory = Eigen::Matrix<float, RacerDubins::STATE_DIM, 500>::Zero();
  RacerDubins::state_array state_der;
  RacerDubins::state_array x;
  x << 0, 1.46919e-6, 0.0140179, 1.09739e-8, -0.000735827;

  for (int i = 0; i < 500; i++)
  {
    RacerDubins::control_array u = control_trajectory.col(i);
    dynamics.computeDynamics(x, u, state_der);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der.allFinite());
    dynamics.updateState(x, state_der, 0.02);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der == RacerDubins::state_array::Zero());
  }
  params.steering_constant = 0.5;
  dynamics.setParams(params);

  x << 0, 1.46919e-6, 0.0140179, 1.09739e-8, -1.0;
  for (int i = 0; i < 500; i++)
  {
    RacerDubins::control_array u = control_trajectory.col(i);
    dynamics.computeDynamics(x, u, state_der);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der.allFinite());
    dynamics.updateState(x, state_der, 0.02);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der == RacerDubins::state_array::Zero());
  }
}

/*
class LinearDummy : public RacerDubins {
public:
  bool computeGrad(const Eigen::Ref<const state_array> & state,
                   const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A,
                   Eigen::Ref<dfdu> B) {
    return false;
  };
};

TEST(RacerDubins, TestComputeGradComputation) {
  Eigen::Matrix<float, RacerDubins::STATE_DIM, RacerDubins::STATE_DIM + RacerDubins::CONTROL_DIM> numeric_jac;
  Eigen::Matrix<float, RacerDubins::STATE_DIM, RacerDubins::STATE_DIM + RacerDubins::CONTROL_DIM> analytic_jac;
  RacerDubins::state_array state;
  state << 1, 2, 3, 4;
  RacerDubins::control_array control;
  control << 5;

  auto analytic_grad_model = RacerDubins();

  RacerDubins::dfdx A_analytic = RacerDubins::dfdx::Zero();
  RacerDubins::dfdu B_analytic = RacerDubins::dfdu::Zero();

  analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);

  auto numerical_grad_model = LinearDummy();

  std::shared_ptr<ModelWrapperDDP<LinearDummy>> ddp_model =
std::make_shared<ModelWrapperDDP<LinearDummy>>(&numerical_grad_model);

  analytic_jac.leftCols<RacerDubins::STATE_DIM>() = A_analytic;
  analytic_jac.rightCols<RacerDubins::CONTROL_DIM>() = B_analytic;
  numeric_jac = ddp_model->df(state, control);

  ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n" << numeric_jac << "\nAnalytic Jacobian\n"
<< analytic_jac;
}

*/
