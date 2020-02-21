#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <dynamics/dynamics.cuh>

#include <dynamics/dynamics_generic_kernel_tests.cuh>

struct DynamicsTesterParams {
  int var_1 = 1;
  int var_2 = 2;
  float4 var_4;
};

template<int STATE_DIM = 1, int CONTROL_DIM = 1>
class DynamicsTester : public MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM> {
public:

    using state_array = typename MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>::state_array;
    using control_array = typename MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>::control_array;

  DynamicsTester(cudaStream_t stream=0)
    : MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>(stream) {}

  DynamicsTester(std::array<float2, CONTROL_DIM> control_rngs, cudaStream_t stream=0)
    : MPPI_internal::Dynamics<DynamicsTester<STATE_DIM, CONTROL_DIM>, DynamicsTesterParams, STATE_DIM, CONTROL_DIM>(control_rngs, stream) {}

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der) {
    state_der(1) = control(0);
  }

  void computeKinematics(const Eigen::Ref<const state_array> &state, Eigen::Ref<state_array> s_der) {
    s_der(0) = state(0) + state(1);
  };

  // TODO must be properly parallelized
  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s = nullptr) {
    state_der[1] = control[0];
  }

  // TODO must be properly parallelized
  __device__ void computeKinematics(float* state, float* state_der) {
    state_der[0] = state[0] + state[1];
  }
};


TEST(Dynamics, BindStream) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  auto tester = DynamicsTester<>(stream);

  EXPECT_EQ(tester.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));

  HANDLE_ERROR(cudaStreamCreate(&stream));

  std::array<float2, 1> tester_ranges {};
  tester = DynamicsTester<>(tester_ranges, stream);

  EXPECT_EQ(tester.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}


TEST(Dynamics, GPUSetupAndCudaFree) {
  DynamicsTester<> tester;
  EXPECT_EQ(tester.model_d_, nullptr);
  EXPECT_EQ(tester.GPUMemStatus_, false);

  tester.GPUSetup();
  EXPECT_NE(tester.model_d_, nullptr);
  EXPECT_EQ(tester.GPUMemStatus_, true);

  tester.freeCudaMem();
  EXPECT_EQ(tester.model_d_, nullptr);
  EXPECT_EQ(tester.GPUMemStatus_, false);
}


TEST(Dynamics, setParamsCPU) {
  DynamicsTester<> tester;
  DynamicsTesterParams params_result = tester.getParams();
  EXPECT_EQ(params_result.var_1, 1);
  EXPECT_EQ(params_result.var_2, 2);

  DynamicsTesterParams params;
  params.var_1 = 10;
  params.var_2 = 20;
  params.var_4.x = 1.5;
  params.var_4.y = 2.5;
  params.var_4.z = 3.5;
  params.var_4.w = 4.5;

  tester.setParams(params);
  params_result = tester.getParams();

  EXPECT_EQ(params_result.var_1, params.var_1);
  EXPECT_EQ(params_result.var_2, params.var_2);
  EXPECT_EQ(params_result.var_4.x, params.var_4.x);
  EXPECT_EQ(params_result.var_4.y, params.var_4.y);
  EXPECT_EQ(params_result.var_4.z, params.var_4.z);
  EXPECT_EQ(params_result.var_4.w, params.var_4.w);
}

TEST(Dynamics, setParamsGPU) {
  DynamicsTester<> tester;
  tester.GPUSetup();
  DynamicsTesterParams params_result = tester.getParams();
  EXPECT_EQ(params_result.var_1, 1);
  EXPECT_EQ(params_result.var_2, 2);

  DynamicsTesterParams params;
  params.var_1 = 10;
  params.var_2 = 20;
  params.var_4.x = 1.5;
  params.var_4.y = 2.5;
  params.var_4.z = 3.5;
  params.var_4.w = 4.5;

  tester.setParams(params);
  launchParameterTestKernel<DynamicsTester<>, DynamicsTesterParams>(tester, params_result);

  EXPECT_EQ(params_result.var_1, params.var_1);
  EXPECT_EQ(params_result.var_2, params.var_2);
  EXPECT_EQ(params_result.var_4.x, params.var_4.x);
  EXPECT_EQ(params_result.var_4.y, params.var_4.y);
  EXPECT_EQ(params_result.var_4.z, params.var_4.z);
  EXPECT_EQ(params_result.var_4.w, params.var_4.w);
}


TEST(Dynamics, ClassConstants) {
  DynamicsTester<> tester;
  EXPECT_EQ(tester.STATE_DIM, 1);
  EXPECT_EQ(tester.CONTROL_DIM, 1);
  EXPECT_EQ(tester.SHARED_MEM_REQUEST_GRD, 1);
  EXPECT_EQ(tester.SHARED_MEM_REQUEST_BLK, 0);

  DynamicsTester<56, 65> tester_2;
  int state_dim = DynamicsTester<56, 65>::STATE_DIM;
  EXPECT_EQ(state_dim, 56);
  int control_dim = DynamicsTester<56, 65>::CONTROL_DIM;
  EXPECT_EQ(control_dim, 65);
  int shared_mem_request_grd = DynamicsTester<56, 65>::SHARED_MEM_REQUEST_GRD;
  EXPECT_EQ(shared_mem_request_grd, 1);
  int shared_mem_request_blk = DynamicsTester<56, 65>::SHARED_MEM_REQUEST_BLK;
  EXPECT_EQ(shared_mem_request_blk, 0);
}

TEST(Dynamics, SetControlRangesDefault) {
  DynamicsTester<> tester;
  auto ranges = tester.getControlRanges();
  EXPECT_FLOAT_EQ(ranges[0].x, -FLT_MAX);
  EXPECT_FLOAT_EQ(ranges[0].y, FLT_MAX);

  DynamicsTester<4, 2> tester_2;
  auto ranges_2 = tester.getControlRanges();
  for(int i = 0; i < ranges_2.size(); i++) {
    EXPECT_FLOAT_EQ(ranges[i].x, -FLT_MAX);
    EXPECT_FLOAT_EQ(ranges[i].y, FLT_MAX);
  }
}

TEST(Dynamics, SetControlRanges) {
  std::array<float2, 1> tester_ranges {};
  tester_ranges[0].x = -2;
  tester_ranges[0].y = 5;
  DynamicsTester<> tester(tester_ranges);
  auto ranges = tester.getControlRanges();
  EXPECT_FLOAT_EQ(ranges[0].x, -2);
  EXPECT_FLOAT_EQ(ranges[0].y, 5);

  std::array<float2, 2> tester_2_ranges {};
  tester_2_ranges[0].x = -5;
  tester_2_ranges[0].y = 6;
  tester_2_ranges[1].x = -10;
  tester_2_ranges[1].y = 20;
  DynamicsTester<4, 2> tester_2(tester_2_ranges);
  auto ranges_2 = tester_2.getControlRanges();
  for(int i = 0; i < ranges_2.size(); i++) {
    EXPECT_FLOAT_EQ(ranges_2[i].x, tester_2_ranges[i].x) << "failed at index: " << i;
    EXPECT_FLOAT_EQ(ranges_2[i].y, tester_2_ranges[i].y) << "failed at index: " << i;
  }
}

TEST(Dynamics, SetControlRangesGPU) {
  std::array<float2, 1> tester_ranges {};
  tester_ranges[0].x = -2;
  tester_ranges[0].y = 5;
  DynamicsTester<> tester(tester_ranges);
  tester.GPUSetup();
  std::array<float2, 1> ranges_result = {};
  launchControlRangesTestKernel<DynamicsTester<>, 1>(tester, ranges_result);
  EXPECT_FLOAT_EQ(ranges_result[0].x, -2);
  EXPECT_FLOAT_EQ(ranges_result[0].y, 5);

  std::array<float2, 2> tester_2_ranges {};
  tester_2_ranges[0].x = -5;
  tester_2_ranges[0].y = 6;
  tester_2_ranges[1].x = -10;
  tester_2_ranges[1].y = 20;
  DynamicsTester<4, 2> tester_2(tester_2_ranges);
  tester_2.GPUSetup();
  std::array<float2, 2> ranges_result_2 = {};
  launchControlRangesTestKernel<DynamicsTester<4, 2>, 2>(tester_2, ranges_result_2);
  for(int i = 0; i < ranges_result_2.size(); i++) {
    EXPECT_FLOAT_EQ(ranges_result_2[i].x, tester_2_ranges[i].x) << "failed at index: " << i;
    EXPECT_FLOAT_EQ(ranges_result_2[i].y, tester_2_ranges[i].y) << "failed at index: " << i;
  }
}


TEST(Dynamics, enforceConstraintsCPU) {
  std::array<float2, 1> tester_ranges {};
  tester_ranges[0].x = -2;
  tester_ranges[0].y = 5;
  DynamicsTester<> tester(tester_ranges);

  DynamicsTester<>::state_array s(1,1);
  DynamicsTester<>::control_array u(1,1);

  u(0) = 100;
  tester.enforceConstraints(s, u);
  EXPECT_FLOAT_EQ(u(0), 5);

  u(0) = -42178;
  tester.enforceConstraints(s, u);
  EXPECT_FLOAT_EQ(u(0), -2);

  u(0) = 2;
  tester.enforceConstraints(s, u);
  EXPECT_FLOAT_EQ(u(0), 2);

  u(0) = -1.5;
  tester.enforceConstraints(s, u);
  EXPECT_FLOAT_EQ(u(0), -1.5);
}

TEST(Dynamics, enforceConstraintsGPU) {
  std::array<float2, 3> tester_ranges {};
  tester_ranges[0].x = -2;
  tester_ranges[0].y = 5;
  tester_ranges[1].x = -6;
  tester_ranges[1].y = 8;
  tester_ranges[2].x = -11;
  tester_ranges[2].y = 16;
  DynamicsTester<1,3> tester(tester_ranges);
  tester.GPUSetup();

  std::vector<std::array<float, 1>> states(4);
  std::vector<std::array<float, 3>> controls(4);

  states[0][0] = 48;
  states[1][0] = 4518;
  states[2][0] = 451;
  states[3][0] = 4516;

  controls[0][0] = 48;
  controls[0][1] = 48;
  controls[0][2] = 48;
  controls[1][0] = -51;
  controls[1][1] = -51;
  controls[1][2] = -51;
  controls[2][0] = 2;
  controls[2][1] = 2;
  controls[2][2] = 2;
  controls[3][0] = -1.5;
  controls[3][1] = -1.5;
  controls[3][2] = -1.5;

  // try a bunch of different y dim
  for(int j = 1; j < 6; j++) {
    states[0][0] = 48;
    states[1][0] = 4518;
    states[2][0] = 451;
    states[3][0] = 4516;

    controls[0][0] = 48;
    controls[0][1] = 48;
    controls[0][2] = 48;
    controls[1][0] = -51;
    controls[1][1] = -51;
    controls[1][2] = -51;
    controls[2][0] = 2;
    controls[2][1] = 2;
    controls[2][2] = 2;
    controls[3][0] = -1.5;
    controls[3][1] = -1.5;
    controls[3][2] = -1.5;

    launchEnforceConstraintTestKernel<DynamicsTester<1, 3>, 1, 3>(tester, states, controls, j);

    EXPECT_FLOAT_EQ(controls[0][0], 5);
    EXPECT_FLOAT_EQ(controls[0][1], 8);
    EXPECT_FLOAT_EQ(controls[0][2], 16);
    EXPECT_FLOAT_EQ(controls[1][0], -2);
    EXPECT_FLOAT_EQ(controls[1][1], -6);
    EXPECT_FLOAT_EQ(controls[1][2], -11);
    EXPECT_FLOAT_EQ(controls[2][0], 2);
    EXPECT_FLOAT_EQ(controls[2][1], 2);
    EXPECT_FLOAT_EQ(controls[2][2], 2);
    EXPECT_FLOAT_EQ(controls[3][0], -1.5);
    EXPECT_FLOAT_EQ(controls[3][1], -1.5);
    EXPECT_FLOAT_EQ(controls[3][2], -1.5);

    EXPECT_FLOAT_EQ(states[0][0], 48);
    EXPECT_FLOAT_EQ(states[1][0], 4518);
    EXPECT_FLOAT_EQ(states[2][0], 451);
    EXPECT_FLOAT_EQ(states[3][0], 4516);
  }
}

TEST(Dynamics, updateStateCPU) {
  DynamicsTester<> tester;
  DynamicsTester<>::state_array s;
  DynamicsTester<>::state_array s_der;

  s(0) = 5;
  s_der(0) = 10;

  tester.updateState(s, s_der, 0.1);

  EXPECT_FLOAT_EQ(s(0), 6);
  EXPECT_FLOAT_EQ(s_der(0), 0);
}

TEST(Dynamics, updateStateGPU) {
  DynamicsTester<4, 1> tester;
  std::vector<std::array<float, 4>> s(1);
  std::vector<std::array<float, 4>> s_der(1);

  s[0][0] = 0;
  s[0][1] = 1;
  s[0][2] = 2;
  s[0][3] = 3;

  s_der[0][0] = 0;
  s_der[0][1] = 1;
  s_der[0][2] = 2;
  s_der[0][3] = 3;

  for(int i = 1; i < 6; i++) {
    s[0][0] = 0;
    s[0][1] = 1;
    s[0][2] = 2;
    s[0][3] = 3;

    s_der[0][0] = 0;
    s_der[0][1] = 1;
    s_der[0][2] = 2;
    s_der[0][3] = 3;

    launchUpdateStateTestKernel<DynamicsTester<4, 1>, 4>(tester, s, s_der, 0.1, i);

    EXPECT_FLOAT_EQ(s[0][0], 0);
    EXPECT_FLOAT_EQ(s[0][1], 1.1);
    EXPECT_FLOAT_EQ(s[0][2], 2.2);
    EXPECT_FLOAT_EQ(s[0][3], 3.3);
    EXPECT_FLOAT_EQ(s_der[0][0], 0);
    EXPECT_FLOAT_EQ(s_der[0][1], 0);
    EXPECT_FLOAT_EQ(s_der[0][2], 0);
    EXPECT_FLOAT_EQ(s_der[0][3], 0);
  }

}

TEST(Dynamics, computeStateDerivCPU) {
  DynamicsTester<2, 1> tester;
  DynamicsTester<2, 1>::state_array s;
  DynamicsTester<2, 1>::state_array s_der;
  DynamicsTester<2, 1>::control_array u;

  s(0) = 5;
  s(1) = 10;
  s_der(0) = 10;
  s_der(1) = 20;
  u(0)=  3;

  tester.computeStateDeriv(s, u, s_der);

  EXPECT_FLOAT_EQ(s(0), 5);
  EXPECT_FLOAT_EQ(s(1), 10);
  EXPECT_FLOAT_EQ(s_der(0), 15);
  EXPECT_FLOAT_EQ(s_der(1), 3);
  EXPECT_FLOAT_EQ(u(0), 3);
}

TEST(Dynamics, computeStateDerivGPU) {
  DynamicsTester<2, 1> tester;
  std::vector<std::array<float, 2>> s(1);
  std::vector<std::array<float, 1>> u(1);
  std::vector<std::array<float, 2>> s_der(1);

  for(int j = 1; j < 6; j++) {
    s[0][0] = 5;
    s[0][1] = 10;
    s_der[0][0] = 10;
    s_der[0][1] = 20;
    u[0][0] = 3;

    launchComputeStateDerivTestKernel<DynamicsTester<2, 1>, 2, 1>(tester, s, u, s_der, j);

    EXPECT_FLOAT_EQ(s[0][0], 5) << "j = " << j;
    EXPECT_FLOAT_EQ(s[0][1], 10) << "j = " << j;
    EXPECT_FLOAT_EQ(s_der[0][0], 15) << "j = " << j;
    EXPECT_FLOAT_EQ(s_der[0][1], 3) << "j = " << j;
    EXPECT_FLOAT_EQ(u[0][0], 3) << "j = " << j;

  }
}

