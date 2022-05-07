#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>
#include <mppi/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <cuda_runtime.h>

class RacerDubinsElevationTest : public ::testing::Test
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

TEST_F(RacerDubinsElevationTest, Template)
{
  auto dynamics = RacerDubinsElevation();
  EXPECT_EQ(7, RacerDubinsElevation::STATE_DIM);
  EXPECT_EQ(2, RacerDubinsElevation::CONTROL_DIM);
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
}

TEST_F(RacerDubinsElevationTest, BindStream)
{
  auto dynamics = RacerDubinsElevation(stream);

  EXPECT_EQ(dynamics.stream_, stream) << "Stream binding failure.";
  EXPECT_NE(dynamics.getTextureHelper(), nullptr);
  EXPECT_EQ(dynamics.getTextureHelper()->stream_, stream);
}

/*
float c_t = 1.3;
float c_b = 2.5;
float c_v = 3.7;
float c_0 = 4.9;
float wheel_base = 0.3;
 */

TEST_F(RacerDubinsElevationTest, ComputeDynamics)
{
  RacerDubinsElevation dynamics = RacerDubinsElevation();
  RacerDubinsParams params = dynamics.getParams();
  RacerDubinsElevation::state_array x = RacerDubinsElevation::state_array::Zero();
  RacerDubinsElevation::control_array u = RacerDubinsElevation::control_array::Zero();

  // computeDynamics should not touch the roll/pitch element
  RacerDubinsElevation::state_array next_x = RacerDubinsElevation::state_array::Ones() * 0.153;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), 0);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << 1, M_PI_2, 0, 3, 0, 0.5, -0.5;
  u << 1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 + 1.3 - 3.7);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_NEAR(next_x(2), 0, 1e-7);
  EXPECT_FLOAT_EQ(next_x(3), 1);
  EXPECT_FLOAT_EQ(next_x(4), 0);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << 1, 0, 0, 3, 0, 0.5, -0.5;
  u << -1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 - 2.5 - 3.7);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), 1);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << -1, 0, 0, 3, 0, 0.5, -0.5;
  u << 1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 + 3.7 + 1.3);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << -1, 0, 0, 3, 0, 0.5, -0.5;
  u << -1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 + 2.5 + 3.7);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << -3, 0, 0, 3, 0, 0.5, -0.5;
  u << -1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 + 2.5 + 3.7 * 3);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), -3);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << 4, 0, 0, 3, 0, 0.5, -0.5;
  u << -1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 - 2.5 - 3.7 * 4);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), 4);
  EXPECT_FLOAT_EQ(next_x(3), 0);
  EXPECT_FLOAT_EQ(next_x(4), 0);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << 1, M_PI, 0, 3, 0, 0.5, -0.5;
  u << 0, 1;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.7);
  EXPECT_FLOAT_EQ(next_x(1), (1 / .3) * tan(0));
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_NEAR(next_x(3), 0, 1e-7);
  EXPECT_FLOAT_EQ(next_x(4), -1 / 2.45);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);

  x << 1, M_PI, 0, 0, 0.5, 0.5, -0.5;
  u << 1, -1;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9 - 3.7 + 1.3);
  EXPECT_FLOAT_EQ(next_x(1), (1 / .3) * tan(0.5));
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_NEAR(next_x(3), 0, 1e-7);
  EXPECT_FLOAT_EQ(next_x(4), 1 / 2.45);
  EXPECT_FLOAT_EQ(next_x(5), 0.153);
  EXPECT_FLOAT_EQ(next_x(6), 0.153);
}

TEST_F(RacerDubinsElevationTest, TestModelGPU)
{
  RacerDubinsElevation dynamics = RacerDubinsElevation();
  dynamics.GPUSetup();
  CudaCheckError();

  Eigen::Matrix<float, RacerDubinsElevation::CONTROL_DIM, 100> control_trajectory;
  control_trajectory = Eigen::Matrix<float, RacerDubinsElevation::CONTROL_DIM, 100>::Random();
  Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM, 100> state_trajectory;
  state_trajectory = Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM, 100>::Random();

  std::vector<std::array<float, 7>> s(100);
  std::vector<std::array<float, 7>> s_der(100);
  // steering, throttle
  std::vector<std::array<float, 2>> u(100);
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
    launchComputeDynamicsTestKernel<RacerDubinsElevation, 7, 2>(dynamics, s, u, s_der, y_dim);
    for (int point = 0; point < 100; point++)
    {
      RacerDubinsElevation::state_array state = state_trajectory.col(point);
      RacerDubinsElevation::control_array control = control_trajectory.col(point);
      RacerDubinsElevation::state_array state_der_cpu = RacerDubinsElevation::state_array::Zero();

      dynamics.computeDynamics(state, control, state_der_cpu);
      for (int dim = 0; dim < RacerDubinsElevation::STATE_DIM; dim++)
      {
        EXPECT_NEAR(state_der_cpu(dim), s_der[point][dim], 1e-5) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_TRUE(isfinite(s_der[point][dim]));
      }
    }
  }

  dynamics.freeCudaMem();
  CudaCheckError();
}

TEST_F(RacerDubinsElevationTest, TestUpdateState)
{
  CudaCheckError();
  RacerDubinsElevation dynamics = RacerDubinsElevation();
  RacerDubinsElevation::state_array state;
  RacerDubinsElevation::state_array state_der;

  // TODO add in the elevation map

  state << 0, 0, 0, 0, 0, -0.5, 0.5;
  state_der << 1, 1, 1, 1, 1, 0, 0;
  dynamics.updateState(state, state_der, 0.1);
  EXPECT_TRUE(state_der == RacerDubinsElevation::state_array::Zero());
  EXPECT_FLOAT_EQ(state(0), 0.1);
  EXPECT_FLOAT_EQ(state(1), 0.1);
  EXPECT_FLOAT_EQ(state(2), 0.1);
  EXPECT_FLOAT_EQ(state(3), 0.1);
  EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 0.1));
  EXPECT_FLOAT_EQ(state(5), 0.0);
  EXPECT_FLOAT_EQ(state(6), 0.0);

  state << 0, M_PI - 0.1, 0, 0, 0, -0.5, 0.5;
  state_der << 1, 1, 1, 1, 1;
  dynamics.updateState(state, state_der, 1.0);
  EXPECT_TRUE(state_der == RacerDubinsElevation::state_array::Zero());
  EXPECT_FLOAT_EQ(state(0), 1.0);
  EXPECT_FLOAT_EQ(state(1), 1.0 - M_PI - 0.1);
  EXPECT_FLOAT_EQ(state(2), 1.0);
  EXPECT_FLOAT_EQ(state(3), 1.0);
  EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 1.0));
  EXPECT_FLOAT_EQ(state(5), 0.0);
  EXPECT_FLOAT_EQ(state(6), 0.0);

  state << 0, -M_PI + 0.1, 0, 0, 0, -0.5, 0.5;
  state_der << 1, -1, 1, 1, 1;
  dynamics.updateState(state, state_der, 1.0);
  EXPECT_TRUE(state_der == RacerDubinsElevation::state_array::Zero());
  EXPECT_FLOAT_EQ(state(0), 1.0);
  EXPECT_FLOAT_EQ(state(1), M_PI + 0.1 - 1.0);
  EXPECT_FLOAT_EQ(state(2), 1.0);
  EXPECT_FLOAT_EQ(state(3), 1.0);
  EXPECT_FLOAT_EQ(state(4), 1.0 + (0 - 1.0) * expf(-0.6 * 1.0));
  EXPECT_FLOAT_EQ(state(5), 0.0);
  EXPECT_FLOAT_EQ(state(6), 0.0);

  CudaCheckError();
}

TEST_F(RacerDubinsElevationTest, TestUpdateStateGPU)
{
  CudaCheckError();
  RacerDubinsElevation dynamics = RacerDubinsElevation();

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  TwoDTextureHelper<float>* helper = dynamics.getTextureHelper();
  helper->setExtent(0, extent);

  std::vector<float> data_vec;
  data_vec.resize(10 * 20);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = i * 1.0f;
  }

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);
  helper->updateRotation(0, new_rot_mat);
  helper->updateOrigin(0, make_float3(1, 2, 3));

  helper->updateTexture(0, data_vec);
  helper->updateResolution(0, 10);
  helper->enableTexture(0);
  helper->copyToDevice(true);

  CudaCheckError();
  dynamics.GPUSetup();
  CudaCheckError();

  Eigen::Matrix<float, RacerDubinsElevation::CONTROL_DIM, 100> control_trajectory;
  control_trajectory = Eigen::Matrix<float, RacerDubinsElevation::CONTROL_DIM, 100>::Random();
  Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM, 100> state_trajectory;
  state_trajectory = Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM, 100>::Random();

  std::vector<std::array<float, 7>> s(100);
  std::vector<std::array<float, 7>> s_der(100);
  // steering, throttle
  std::vector<std::array<float, 2>> u(100);

  RacerDubinsElevation::state_array state;
  RacerDubinsElevation::control_array control;
  RacerDubinsElevation::state_array state_der_cpu = RacerDubinsElevation::state_array::Zero();

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for (int y_dim = 1; y_dim <= 10; y_dim++)
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

    launchComputeStateDerivTestKernel<RacerDubinsElevation, RacerDubinsElevation::STATE_DIM,
                                      RacerDubinsElevation::CONTROL_DIM>(dynamics, s, u, s_der, y_dim);
    launchUpdateStateTestKernel<RacerDubinsElevation, RacerDubinsElevation::STATE_DIM>(dynamics, s, s_der, 0.1f, y_dim);
    for (int point = 0; point < 100; point++)
    {
      RacerDubinsElevation::state_array state = state_trajectory.col(point);
      RacerDubinsElevation::control_array control = control_trajectory.col(point);
      RacerDubinsElevation::state_array state_der_cpu = RacerDubinsElevation::state_array::Zero();

      dynamics.computeDynamics(state, control, state_der_cpu);
      dynamics.updateState(state, state_der_cpu, 0.1f);
      for (int dim = 0; dim < RacerDubinsElevation::STATE_DIM; dim++)
      {
        EXPECT_FLOAT_EQ(state_der_cpu(dim), s_der[point][dim]) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_NEAR(state(dim), s[point][dim], 1e-4) << "at index " << point << " with y_dim " << y_dim;
        EXPECT_TRUE(isfinite(s[point][dim]));
      }
    }
  }
  dynamics.freeCudaMem();
}

TEST_F(RacerDubinsElevationTest, ComputeStateTrajectoryFiniteTest)
{
  RacerDubinsElevation dynamics = RacerDubinsElevation();
  RacerDubinsParams params;
  params.c_t = 3.0;
  params.c_b = 0.2;
  params.c_v = 0.2;
  params.c_0 = 0.2;
  params.wheel_base = 3.0;
  params.steering_constant = 1.0;
  dynamics.setParams(params);

  Eigen::Matrix<float, RacerDubinsElevation::CONTROL_DIM, 500> control_trajectory;
  control_trajectory = Eigen::Matrix<float, RacerDubinsElevation::CONTROL_DIM, 500>::Zero();
  Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM, 500> state_trajectory;
  state_trajectory = Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM, 500>::Zero();
  RacerDubinsElevation::state_array state_der;
  RacerDubinsElevation::state_array x;
  x << 0, 1.46919e-6, 0.0140179, 1.09739e-8, -0.000735827;

  for (int i = 0; i < 500; i++)
  {
    RacerDubinsElevation::control_array u = control_trajectory.col(i);
    dynamics.computeDynamics(x, u, state_der);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der.allFinite());
    dynamics.updateState(x, state_der, 0.02);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der == RacerDubinsElevation::state_array::Zero());
  }
  params.steering_constant = 0.5;
  dynamics.setParams(params);

  x << 0, 1.46919e-6, 0.0140179, 1.09739e-8, -1.0;
  for (int i = 0; i < 500; i++)
  {
    RacerDubinsElevation::control_array u = control_trajectory.col(i);
    dynamics.computeDynamics(x, u, state_der);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der.allFinite());
    dynamics.updateState(x, state_der, 0.02);
    EXPECT_TRUE(x.allFinite());
    EXPECT_TRUE(u.allFinite());
    EXPECT_TRUE(state_der == RacerDubinsElevation::state_array::Zero());
  }
}

/*
class LinearDummy : public RacerDubinsElevation {
public:
  bool computeGrad(const Eigen::Ref<const state_array> & state,
                   const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A,
                   Eigen::Ref<dfdu> B) {
    return false;
  };
};

TEST_F(RacerDubinsElevationTest, TestComputeGradComputation) {
  Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM, RacerDubinsElevation::STATE_DIM +
RacerDubinsElevation::CONTROL_DIM> numeric_jac; Eigen::Matrix<float, RacerDubinsElevation::STATE_DIM,
RacerDubinsElevation::STATE_DIM + RacerDubinsElevation::CONTROL_DIM> analytic_jac; RacerDubinsElevation::state_array
state; state << 1, 2, 3, 4; RacerDubinsElevation::control_array control; control << 5;

  auto analytic_grad_model = RacerDubinsElevation();

  RacerDubinsElevation::dfdx A_analytic = RacerDubinsElevation::dfdx::Zero();
  RacerDubinsElevation::dfdu B_analytic = RacerDubinsElevation::dfdu::Zero();

  analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);

  auto numerical_grad_model = LinearDummy();

  std::shared_ptr<ModelWrapperDDP<LinearDummy>> ddp_model =
std::make_shared<ModelWrapperDDP<LinearDummy>>(&numerical_grad_model);

  analytic_jac.leftCols<RacerDubinsElevation::STATE_DIM>() = A_analytic;
  analytic_jac.rightCols<RacerDubinsElevation::CONTROL_DIM>() = B_analytic;
  numeric_jac = ddp_model->df(state, control);

  ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n" << numeric_jac << "\nAnalytic Jacobian\n"
<< analytic_jac;
}

*/
