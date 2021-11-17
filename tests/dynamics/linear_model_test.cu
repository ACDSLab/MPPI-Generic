#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mppi/dynamics/linear/linear.cuh>
#include <mppi/dynamics/dynamics_generic_kernel_tests.cuh>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <cuda_runtime.h>

TEST(LinearModel, Template) {
  auto dynamics = LinearModel();
  EXPECT_EQ(4, LinearModel::STATE_DIM);
  EXPECT_EQ(4, LinearModel::CONTROL_DIM);
}

TEST(LinearModel, BindStream) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  auto dynamics = LinearModel(stream);

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

TEST(LinearModel, ComputeModel) {
  LinearModel dynamics = LinearModel();
  LinearModelParams params = dynamics.getParams();
  LinearModel::state_array x = LinearModel::state_array::Zero();
  LinearModel::control_array u = LinearModel::state_array::Zero();

  LinearModel::state_array next_x;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), 0);
  EXPECT_FLOAT_EQ(next_x(3), 0);

  x << 1, M_PI_2, 0, 3;
  u << 1, 0, 0, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9+1.3);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_NEAR(next_x(2), 0, 1e-7);
  EXPECT_FLOAT_EQ(next_x(3), 1);

  x << 1, 0, 0, 3;
  u << 0, 1, 0, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9-2.5);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), 1);
  EXPECT_FLOAT_EQ(next_x(3), 0);

  x << 1, M_PI, 0, 3;
  u << 0, 0, 1, 0;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9-3.7);
  EXPECT_FLOAT_EQ(next_x(1), 0);
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_NEAR(next_x(3), 0, 1e-7);

  x << 1, M_PI, 0, 3;
  u << 0, 0, 0, 1;
  dynamics.computeDynamics(x, u, next_x);
  EXPECT_FLOAT_EQ(next_x(0), 4.9);
  EXPECT_FLOAT_EQ(next_x(1), (1/.3)*tan(1));
  EXPECT_FLOAT_EQ(next_x(2), -1);
  EXPECT_NEAR(next_x(3), 0, 1e-7);
}

TEST(LinearModel, TestModelGPU) {
  LinearModel dynamics = LinearModel();
  dynamics.GPUSetup();

  LinearModel::state_array state;
  state(0) = 0.5;
  state(1) = 0.7;
  state(2) = M_PI_4;
  LinearModel::control_array control;
  control(0) = 3.0;
  control(1) = 2.0;

  std::vector<std::array<float, 4>> s(1);
  for(int dim = 0; dim < 4; dim++) {
    s[0][dim] = state(dim);
  }
  std::vector<std::array<float, 4>> s_der(1);
  // steering, throttle
  std::vector<std::array<float, 4>> u(1);
  for(int dim = 0; dim < 4; dim++) {
    u[0][dim] = control(dim);
  }

  // These variables will be changed so initialized to the right size only
  LinearModel::state_array state_der_cpu = LinearModel::state_array::Zero();
  dynamics.computeDynamics(state, control, state_der_cpu);

  // Run dynamics on dynamicsU
  // Run dynamics on GPU
  for(int point = 0; point < 100; point++) {
    for(int y_dim = 1; y_dim <= 4; y_dim++) {
      launchComputeDynamicsTestKernel<LinearModel, 4, 4>(dynamics, s, u, s_der, y_dim);
      for(int dim = 0; dim < 4; dim++) {
        EXPECT_FLOAT_EQ(state_der_cpu(dim), s_der[0][dim]);
      }
    }
    state = LinearModel::state_array ::Random();
    control = LinearModel::control_array ::Random();
    dynamics.computeDynamics(state, control, state_der_cpu);
    for(int dim = 0; dim < 4; dim++) {
      s[0][dim] = state(dim);
    }
    for(int dim = 0; dim < 4; dim++) {
      u[0][dim] = control(dim);
    }

  }

  dynamics.freeCudaMem();
}

/*
TEST(LinearModel, TestUpdateStateGPU) {
  LinearModel dynamics = LinearModel();
  dynamics.GPUSetup();

  LinearModel::state_array state;
  state(0) = 0.5;
  state(1) = 0.7;
  state(2) = M_PI;
  LinearModel::control_array control;
  control(0) = 3.0;
  control(1) = 2.0;

  std::vector<std::array<float, 3>> s(1);
  for(int dim = 0; dim < 3; dim++) {
    s[0][dim] = state(dim);
  }
  std::vector<std::array<float, 3>> s_der(1);
  // steering, throttle
  std::vector<std::array<float, 2>> u(1);
  for(int dim = 0; dim < 2; dim++) {
    u[0][dim] = control(dim);
  }

  // These variables will be changed so initialized to the right size only
  LinearModel::state_array state_der_cpu = LinearModel::state_array::Zero();

  // Run dynamics on dynamicsU
  dynamics.computeDynamics(state, control, state_der_cpu);
  dynamics.updateState(state, state_der_cpu, 0.1f);
  // Run dynamics on GPU
  for(int y_dim = 1; y_dim <= 3; y_dim++) {
    launchComputeStateDerivTestKernel<LinearModel, 3, 2>(dynamics, s, u, s_der, y_dim);
    launchUpdateStateTestKernel<LinearModel, 3>(dynamics, s, s_der, 0.1f, y_dim);
    for(int dim = 0; dim < 3; dim++) {
      EXPECT_FLOAT_EQ(state_der_cpu(dim), s_der[0][dim]);
    }
  }
  dynamics.freeCudaMem();
}

class LinearDummy : public LinearModel {
public:
  bool computeGrad(const Eigen::Ref<const state_array> & state,
                   const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A,
                   Eigen::Ref<dfdu> B) {
    return false;
  };
};

TEST(LinearModel, TestComputeGradComputation) {
  Eigen::Matrix<float, LinearModel::STATE_DIM, LinearModel::STATE_DIM + LinearModel::CONTROL_DIM> numeric_jac;
  Eigen::Matrix<float, LinearModel::STATE_DIM, LinearModel::STATE_DIM + LinearModel::CONTROL_DIM> analytic_jac;
  LinearModel::state_array state;
  state << 1, 2, 3, 4;
  LinearModel::control_array control;
  control << 5;

  auto analytic_grad_model = LinearModel();

  LinearModel::dfdx A_analytic = LinearModel::dfdx::Zero();
  LinearModel::dfdu B_analytic = LinearModel::dfdu::Zero();

  analytic_grad_model.computeGrad(state, control, A_analytic, B_analytic);

  auto numerical_grad_model = LinearDummy();

  std::shared_ptr<ModelWrapperDDP<LinearDummy>> ddp_model = std::make_shared<ModelWrapperDDP<LinearDummy>>(&numerical_grad_model);

  analytic_jac.leftCols<LinearModel::STATE_DIM>() = A_analytic;
  analytic_jac.rightCols<LinearModel::CONTROL_DIM>() = B_analytic;
  numeric_jac = ddp_model->df(state, control);

  ASSERT_LT((numeric_jac - analytic_jac).norm(), 1e-3) << "Numeric Jacobian\n" << numeric_jac << "\nAnalytic Jacobian\n" << analytic_jac;
}

*/
