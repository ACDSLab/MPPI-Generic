#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mppi/controllers/controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>
#include <mppi/utils/test_helper.h>
#include <mppi_test/mock_classes/mock_costs.h>
#include <mppi_test/mock_classes/mock_dynamics.h>

#include <algorithm>
#include <numeric>
#include <random>

static const int number_rollouts = 1200;
static const int NUM_TIMESTEPS = 100;
using FEEDBACK_T = DDPFeedback<MockDynamics>;
const dim3 rolloutDim(1, 2, 1);
using SAMPLER_T = mppi::sampling_distributions::GaussianDistribution<MockDynamics::DYN_PARAMS_T>;

class TestController : public Controller<MockDynamics, MockCost, FEEDBACK_T, SAMPLER_T>
{
public:
  typedef Controller<MockDynamics, MockCost, FEEDBACK_T, SAMPLER_T> PARENT_CLASS;
  using PARAMS_T = PARENT_CLASS::TEMPLATED_PARAMS;

  TestController(MockDynamics* model, MockCost* cost, FEEDBACK_T* fb_controller, SAMPLER_T* sampler, float dt,
                 int max_iter, float lambda, float alpha, int num_timesteps, int num_rollouts,
                 const Eigen::Ref<const control_trajectory>& init_control_traj =
                     control_trajectory::Zero(MockDynamics::CONTROL_DIM, 1),
                 cudaStream_t stream = nullptr)
    : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, num_rollouts,
                   init_control_traj, stream)
  {
    // Allocate CUDA memory for the controller
    allocateCUDAMemoryHelper(1);
  }

  TestController(MockDynamics* model, MockCost* cost, FEEDBACK_T* fb_controller, SAMPLER_T* sampler, PARAMS_T& params,
                 cudaStream_t stream = nullptr)
    : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
  {
    // Allocate CUDA memory for the controller
    allocateCUDAMemoryHelper(1);
  }

  virtual void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride) override
  {
  }

  virtual void calculateSampledStateTrajectories() override{};

  void computeControl(const Eigen::Ref<const state_array>& state,
                      const std::array<control_trajectory, number_rollouts> noise)
  {
    int trajectory_size = control_trajectory().size();
    for (int i = 0; i < this->getNumRollouts(); i++)
    {
      HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getControlSample(i, 0, 0), noise[i].data(),
                                   sizeof(float) * trajectory_size, cudaMemcpyHostToDevice, stream_));
    }
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
    // Normally rolloutKernel would be called here and would transform
    //  control_noise_d_ from u to u + noise

    // Instead we just get back noise in this test
    this->copySampledControlFromDevice();
  }

  virtual void slideControlSequence(int steps) override
  {
  }

  cudaStream_t getStream()
  {
    return stream_;
  }
};

class ControllerTests : public ::testing::Test
{
protected:
  void SetUp() override
  {
    mockDynamics = new MockDynamics();
    mockCost = new MockCost();
    mockFeedback = new FEEDBACK_T(mockDynamics, dt);
    sampler = new SAMPLER_T();
    HANDLE_ERROR(cudaStreamCreate(&stream));
    MockDynamics tmp_dynamics();

    // expect double check rebind
    EXPECT_CALL(*mockCost, bindToStream(testing::_)).Times(1);
    EXPECT_CALL(*mockDynamics, bindToStream(testing::_)).Times(1);
    // EXPECT_CALL(mockFeedback, bindToStream(stream)).Times(1);

    // expect GPU setup called again
    EXPECT_CALL(*mockCost, GPUSetup()).Times(1);
    EXPECT_CALL(*mockDynamics, GPUSetup()).Times(1);
    // EXPECT_CALL(mockFeedback, GPUSetup()).Times(1);

    controller = new TestController(mockDynamics, mockCost, mockFeedback, sampler, dt, max_iter, lambda, alpha,
                                    NUM_TIMESTEPS, number_rollouts);
    auto controller_params = controller->getParams();
    controller_params.dynamics_rollout_dim_ = rolloutDim;
    controller_params.cost_rollout_dim_ = rolloutDim;
    controller->setParams(controller_params);
  }
  void TearDown() override
  {
    EXPECT_CALL(*mockDynamics, freeCudaMem()).Times(1);
    EXPECT_CALL(*mockCost, freeCudaMem()).Times(1);
    delete controller;
    delete mockDynamics;
    delete mockCost;
    delete mockFeedback;
    delete sampler;
    HANDLE_ERROR(cudaStreamDestroy(stream));
  }

  MockDynamics* mockDynamics;
  MockCost* mockCost;
  FEEDBACK_T* mockFeedback;
  SAMPLER_T* sampler;
  TestController* controller;

  float dt = 0.1;
  int max_iter = 1;
  float lambda = 1.2;
  float alpha = 0.1;
  cudaStream_t stream;
};

TEST_F(ControllerTests, ConstructorDestructor)
{
  int num_timesteps = 10;

  TestController::control_trajectory init_control_trajectory =
      TestController::control_trajectory::Ones(MockDynamics::CONTROL_DIM, num_timesteps);

  // expect double check rebind
  EXPECT_CALL(*mockCost, bindToStream(stream)).Times(1);
  EXPECT_CALL(*mockDynamics, bindToStream(stream)).Times(1);
  // // EXPECT_CALL(mockFeedback, bindToStream(stream)).Times(1);

  // // expect GPU setup called again
  EXPECT_CALL(*mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(*mockDynamics, GPUSetup()).Times(1);
  // EXPECT_CALL(mockFeedback, GPUSetup()).Times(1);
  TestController* controller_test =
      new TestController(mockDynamics, mockCost, mockFeedback, sampler, dt, max_iter, lambda, alpha, num_timesteps,
                         number_rollouts, init_control_trajectory, stream);

  EXPECT_EQ(controller_test->model_, mockDynamics);
  EXPECT_EQ(controller_test->cost_, mockCost);
  EXPECT_EQ(controller_test->getDt(), dt);
  EXPECT_EQ(controller_test->getNumIters(), max_iter);
  EXPECT_EQ(controller_test->getLambda(), lambda);
  EXPECT_EQ(controller_test->getAlpha(), alpha);
  EXPECT_EQ(controller_test->getNumTimesteps(), num_timesteps);
  EXPECT_EQ(controller_test->getNumRollouts(), number_rollouts);
  EXPECT_EQ(controller_test->getStream(), stream);
  EXPECT_EQ(controller_test->getFeedbackEnabled(), false);

  // TODO check that a random seed was set and stream was set
  // EXPECT_NE(controller_test->getRandomSeed(), 0);

  // TODO check for correct defaults
  EXPECT_CALL(*mockDynamics, freeCudaMem()).Times(1);
  EXPECT_CALL(*mockCost, freeCudaMem()).Times(1);
  delete controller_test;
}

TEST_F(ControllerTests, ArgumentConstructorControlTrajectories)
{
  int num_timesteps = 10;

  TestController::control_trajectory init_control_trajectory =
      TestController::control_trajectory::Random(MockDynamics::CONTROL_DIM, 1);

  // expect double check rebind
  EXPECT_CALL(*mockCost, bindToStream(stream)).Times(1);
  EXPECT_CALL(*mockDynamics, bindToStream(stream)).Times(1);
  // // EXPECT_CALL(mockFeedback, bindToStream(stream)).Times(1);

  // // expect GPU setup called again
  EXPECT_CALL(*mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(*mockDynamics, GPUSetup()).Times(1);
  TestController* controller_test =
      new TestController(mockDynamics, mockCost, mockFeedback, sampler, dt, max_iter, lambda, alpha, num_timesteps,
                         number_rollouts, init_control_trajectory, stream);

  auto new_controller_params = controller_test->getParams();
  TestController::control_array model_zero_control = mockDynamics->getZeroControl();
  TestController::control_array expected_end_control =
      (init_control_trajectory.col(0).array() - model_zero_control.array()) *
          new_controller_params.slide_control_scale_.array() +
      model_zero_control.array();

  // Check control trajectories
  for (int t = 0; t < num_timesteps; t++)
  {
    if (t < init_control_trajectory.cols())
    {
      EXPECT_FLOAT_EQ(fabsf((new_controller_params.init_control_traj_.col(t) - init_control_trajectory.col(t)).sum()),
                      0.0f)
          << "Params init control at time " << t << " did not match initial controls passed into constructor";
    }
    else
    {
      EXPECT_NEAR(fabsf((new_controller_params.init_control_traj_.col(t) - expected_end_control).sum()), 0.0f, 1e-10f)
          << "Params init control at time " << t << " did not match initial controls passed into constructor";
    }
  }
  TestController::control_trajectory new_control_trajectory = controller_test->getControlSeq();
  for (int t = 0; t < num_timesteps; t++)
  {
    if (t < init_control_trajectory.cols())
    {
      EXPECT_NEAR(fabsf((new_control_trajectory.col(t) - init_control_trajectory.col(t)).sum()), 0.0f, 1e-10f)
          << "Control vector at time " << t << " did not match initial controls passed into constructor";
    }
    else
    {
      EXPECT_NEAR(fabsf((new_control_trajectory.col(t) - expected_end_control).sum()), 0.0f, 1e-10f)
          << "Control vector at time " << t << " did not match initial controls passed into constructor";
    }
  }
  EXPECT_CALL(*mockDynamics, freeCudaMem()).Times(1);
  EXPECT_CALL(*mockCost, freeCudaMem()).Times(1);
  delete controller_test;
}

TEST_F(ControllerTests, ParamBasedConstructor)
{
  int num_timesteps = 10;
  TestController::TEMPLATED_PARAMS controller_params;
  controller_params.num_timesteps_ = num_timesteps;
  controller_params.num_rollouts_ = number_rollouts;
  controller_params.dt_ = dt;
  controller_params.num_iters_ = max_iter;
  controller_params.lambda_ = lambda;
  controller_params.alpha_ = alpha;
  controller_params.init_control_traj_ =
      TestController::control_trajectory::Ones(MockDynamics::CONTROL_DIM, num_timesteps);

  // expect double check rebind
  EXPECT_CALL(*mockCost, bindToStream(stream)).Times(1);
  EXPECT_CALL(*mockDynamics, bindToStream(stream)).Times(1);
  // // EXPECT_CALL(mockFeedback, bindToStream(stream)).Times(1);

  // // expect GPU setup called again
  EXPECT_CALL(*mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(*mockDynamics, GPUSetup()).Times(1);
  // EXPECT_CALL(mockFeedback, GPUSetup()).Times(1);
  TestController* controller_test =
      new TestController(mockDynamics, mockCost, mockFeedback, sampler, controller_params, stream);

  EXPECT_EQ(controller_test->model_, mockDynamics);
  EXPECT_EQ(controller_test->cost_, mockCost);
  EXPECT_EQ(controller_test->getDt(), dt);
  EXPECT_EQ(controller_test->getNumIters(), max_iter);
  EXPECT_EQ(controller_test->getLambda(), lambda);
  EXPECT_EQ(controller_test->getAlpha(), alpha);
  EXPECT_EQ(controller_test->getNumTimesteps(), num_timesteps);
  EXPECT_EQ(controller_test->getNumRollouts(), number_rollouts);
  // EXPECT_EQ(controller_test->getControlSeq(), controller_params.init_control_traj_);
  EXPECT_EQ(controller_test->getStream(), stream);
  EXPECT_EQ(controller_test->getFeedbackEnabled(), false);

  // TODO check that a random seed was set and stream was set
  // EXPECT_NE(controller_test->getRandomSeed(), 0);

  // TODO check for correct defaults
  EXPECT_CALL(*mockDynamics, freeCudaMem()).Times(1);
  EXPECT_CALL(*mockCost, freeCudaMem()).Times(1);
  delete controller_test;
}

TEST_F(ControllerTests, ParamConstructorControlTrajectories)
{
  int num_timesteps = 10;
  TestController::TEMPLATED_PARAMS controller_params;
  controller_params.num_timesteps_ = num_timesteps;
  controller_params.num_rollouts_ = number_rollouts;
  controller_params.dt_ = dt;
  controller_params.num_iters_ = max_iter;
  controller_params.lambda_ = lambda;
  controller_params.alpha_ = alpha;
  controller_params.init_control_traj_ = TestController::control_trajectory::Random(MockDynamics::CONTROL_DIM, 1);
  // fill in empty portions of control trajectories with last control value
  controller_params.slide_control_scale_ = TestController::control_array::Ones();

  // expect double check rebind
  EXPECT_CALL(*mockCost, bindToStream(stream)).Times(1);
  EXPECT_CALL(*mockDynamics, bindToStream(stream)).Times(1);

  // // expect GPU setup called again
  EXPECT_CALL(*mockCost, GPUSetup()).Times(1);
  EXPECT_CALL(*mockDynamics, GPUSetup()).Times(1);
  TestController* controller_test =
      new TestController(mockDynamics, mockCost, mockFeedback, sampler, controller_params, stream);

  auto new_controller_params = controller_test->getParams();
  // Check control trajectories
  for (int t = 0; t < num_timesteps; t++)
  {
    EXPECT_FLOAT_EQ(
        fabsf((new_controller_params.init_control_traj_.col(t) - controller_params.init_control_traj_.col(0)).sum()),
        0.0f)
        << "Params init control at time " << t << " did not match initial controls passed into constructor";
  }
  auto new_control_trajectory = controller_test->getControlSeq();
  for (int t = 0; t < num_timesteps; t++)
  {
    EXPECT_FLOAT_EQ(fabsf((new_control_trajectory.col(t) - controller_params.init_control_traj_.col(0)).sum()), 0.0f)
        << "Control vector at time " << t << " did not match initial controls passed into constructor";
  }
  EXPECT_CALL(*mockDynamics, freeCudaMem()).Times(1);
  EXPECT_CALL(*mockCost, freeCudaMem()).Times(1);
  delete controller_test;
}

TEST_F(ControllerTests, setNumTimesteps)
{
  controller->setNumTimesteps(10);
  auto control_traj = controller->getControlSeq();
  auto target_state_traj = controller->getTargetStateSeq();
  auto actual_state_traj = controller->getActualStateSeq();
  auto feedback_state_traj = controller->getFeedbackPropagatedStateSeq();
  auto target_output_traj = controller->getTargetOutputSeq();
  auto actual_output_traj = controller->getActualOutputSeq();
  auto controller_params = controller->getParams();
  auto sampler_params = controller->getSamplingParams();

  EXPECT_EQ(controller->getNumTimesteps(), 10);
  EXPECT_EQ(control_traj.cols(), 10);
  EXPECT_EQ(target_state_traj.cols(), 10);
  EXPECT_EQ(actual_state_traj.cols(), 10);
  EXPECT_EQ(feedback_state_traj.cols(), 10);
  EXPECT_EQ(target_output_traj.cols(), 10);
  EXPECT_EQ(actual_output_traj.cols(), 10);
  EXPECT_EQ(controller_params.init_control_traj_.cols(), 10);
  EXPECT_EQ(sampler_params.num_timesteps, 10);

  controller->setNumTimesteps(1000);
  control_traj = controller->getControlSeq();
  target_state_traj = controller->getTargetStateSeq();
  actual_state_traj = controller->getActualStateSeq();
  feedback_state_traj = controller->getFeedbackPropagatedStateSeq();
  target_output_traj = controller->getTargetOutputSeq();
  actual_output_traj = controller->getActualOutputSeq();
  controller_params = controller->getParams();
  sampler_params = controller->getSamplingParams();

  EXPECT_EQ(controller->getNumTimesteps(), 1000);
  EXPECT_EQ(control_traj.cols(), 1000);
  EXPECT_EQ(target_state_traj.cols(), 1000);
  EXPECT_EQ(actual_state_traj.cols(), 1000);
  EXPECT_EQ(feedback_state_traj.cols(), 1000);
  EXPECT_EQ(target_output_traj.cols(), 1000);
  EXPECT_EQ(actual_output_traj.cols(), 1000);
  EXPECT_EQ(controller_params.init_control_traj_.cols(), 1000);
  EXPECT_EQ(sampler_params.num_timesteps, 1000);
}

TEST_F(ControllerTests, setNumRollouts)
{
  controller->setNumRollouts(10);
  auto sampler_params = controller->getSamplingParams();
  auto cost_trajectory = controller->getSampledCostSeq();

  EXPECT_EQ(controller->getNumRollouts(), 10);
  EXPECT_EQ(cost_trajectory.rows(), 10);
  EXPECT_EQ(sampler_params.num_rollouts, 10);

  controller->setNumRollouts(1000);
  sampler_params = controller->getSamplingParams();
  cost_trajectory = controller->getSampledCostSeq();

  EXPECT_EQ(controller->getNumRollouts(), 1000);
  EXPECT_EQ(cost_trajectory.rows(), 1000);
  EXPECT_EQ(sampler_params.num_rollouts, 1000);
}

TEST_F(ControllerTests, smoothControlTrajectory)
{
  TestController::control_trajectory u =
      TestController::control_trajectory::Zero(MockDynamics::CONTROL_DIM, NUM_TIMESTEPS);
  u.col(0) = TestController::control_array::Ones();
  u.col(1) = 2.0 * TestController::control_array::Ones();
  Eigen::Matrix<float, MockDynamics::CONTROL_DIM, 2> control_history =
      Eigen::Matrix<float, MockDynamics::CONTROL_DIM, 2>::Zero();

  // smooth using only one value from the control trajectory
  controller->setNumTimesteps(1);
  controller->smoothControlTrajectoryHelper(u, control_history);
  for (int i = 0; i < MockDynamics::CONTROL_DIM; i++)
  {
    EXPECT_FLOAT_EQ(u(i, 0), (1.0 * 17 + 1.0 * 12 + 1.0 * -3) / 35.0) << "i = " << i;
  }
  // Reset trajectory and smooth over two steps
  u.col(0) = TestController::control_array::Ones();
  controller->setNumTimesteps(2);
  controller->smoothControlTrajectoryHelper(u, control_history);
  for (int i = 0; i < MockDynamics::CONTROL_DIM; i++)
  {
    EXPECT_FLOAT_EQ(u(i, 0), (1.0 * 17 + 2.0 * 12 + 2.0 * -3) / 35.0) << "i = " << i;
    EXPECT_FLOAT_EQ(u(i, 1), (1.0 * 12 + 2.0 * 17 + 2.0 * 12 + 2.0 * -3) / 35.0) << "i = " << i;
  }
}

TEST_F(ControllerTests, slideControlSequenceHelper)
{
  TestController::control_trajectory u =
      TestController::control_trajectory::Zero(MockDynamics::CONTROL_DIM, NUM_TIMESTEPS);
  for (int i = 0; i < controller->getNumTimesteps(); i++)
  {
    TestController::control_array control = TestController::control_array::Ones();
    control = control * i;
    u.col(i) = control.transpose();
  }

  controller->slideControlSequenceHelper(1, u);
  for (int i = 0; i < controller->getNumTimesteps(); i++)
  {
    for (int j = 0; j < MockDynamics::CONTROL_DIM; j++)
    {
      int val = std::min(i + 1, controller->getNumTimesteps() - 1);
      if (i + 1 > controller->getNumTimesteps() - 1)
      {
        EXPECT_FLOAT_EQ(u(j, i), 0);
      }
      else
      {
        EXPECT_FLOAT_EQ(u(j, i), val);
      }
    }
  }

  controller->slideControlSequenceHelper(10, u);
  for (int i = 0; i < controller->getNumTimesteps(); i++)
  {
    for (int j = 0; j < MockDynamics::CONTROL_DIM; j++)
    {
      int val = std::min(i + 11, controller->getNumTimesteps() - 1);
      if (i + 10 > controller->getNumTimesteps() - 2)
      {
        EXPECT_FLOAT_EQ(u(j, i), 0);
      }
      else
      {
        EXPECT_FLOAT_EQ(u(j, i), val);
      }
    }
  }
}

TEST_F(ControllerTests, computeStateTrajectoryHelper)
{
  TestController::state_array x = TestController::state_array::Ones();
  TestController::state_array xdot = TestController::state_array::Ones();
  EXPECT_CALL(*mockDynamics, step(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, dt))
      .Times(controller->getNumTimesteps() - 1);

  TestController::state_trajectory result =
      TestController::state_trajectory::Ones(MockDynamics::STATE_DIM, controller->getNumTimesteps());
  TestController::control_trajectory u =
      TestController::control_trajectory::Zero(MockDynamics::CONTROL_DIM, controller->getNumTimesteps());
  EXPECT_CALL(*mockDynamics, enforceConstraints(testing::_, testing::_)).Times(controller->getNumTimesteps() - 1);
  controller->computeStateTrajectoryHelper(result, x, u);

  // TODO: Figure out if we can actually check output if output is not part of input
  // for (int i = 0; i < controller->getNumTimesteps(); i++)
  // {
  //   for (int j = 0; j < MockDynamics::STATE_DIM; j++)
  //   {
  //     EXPECT_FLOAT_EQ(result(j, i), 1.0) << "t: " << i << ", s_dim: " << j << "\n";
  //   }
  // }
}

TEST_F(ControllerTests, interpolateControl)
{
  TestController::control_trajectory traj =
      TestController::control_trajectory::Zero(MockDynamics::CONTROL_DIM, NUM_TIMESTEPS);
  for (int i = 0; i < controller->getNumTimesteps(); i++)
  {
    traj.col(i) = TestController::control_array::Ones() * i;
  }

  for (double i = 0; i < controller->getNumTimesteps() - 1; i += 0.25)
  {
    TestController::control_array result = controller->interpolateControls(i * controller->getDt(), traj);
    EXPECT_FLOAT_EQ(result(0), i) << i;
  }
}

TEST_F(ControllerTests, interpolateFeedback)
{
  controller->initFeedback();
  auto fb_state = controller->getFeedbackState();
  for (int i = 0; i < fb_state.size(); i++)
  {
    fb_state.getFeedbackGainPtr()[i] = i;
  }

  TestController::state_trajectory s_traj =
      TestController::state_trajectory::Zero(MockDynamics::STATE_DIM, controller->getNumTimesteps());

  TestController::state_array state = TestController::state_array::Ones();
  for (double i = 0; i < controller->getNumTimesteps() - 1; i += 0.25)
  {
    TestController::state_array interpolated_state = controller->interpolateState(s_traj, i * controller->getDt());
    TestController::control_array result =
        controller->interpolateFeedback(state, interpolated_state, i * controller->getDt(), fb_state);
    EXPECT_FLOAT_EQ(result(0), i);
  }
}

TEST_F(ControllerTests, getCurrentControlTest)
{
  EXPECT_CALL(*mockDynamics, enforceConstraints(testing::_, testing::_)).Times(4 * (controller->getNumTimesteps() - 1));

  TestController::control_trajectory traj = controller->getControlSeq();
  controller->initFeedback();
  auto fb_state = controller->getFeedbackState();
  for (int i = 0; i < controller->getNumTimesteps(); i++)
  {
    for (int j = 0; j < MockDynamics::STATE_DIM * MockDynamics::CONTROL_DIM; j++)
    {
      int i_index = i * MockDynamics::STATE_DIM * MockDynamics::CONTROL_DIM;
      fb_state.getFeedbackGainPtr()[i_index + j] = i_index + j;
    }
    traj.col(i) = TestController::control_array::Ones() * i;
  }

  TestController::state_trajectory s_traj =
      TestController::state_trajectory::Zero(MockDynamics::STATE_DIM, controller->getNumTimesteps());

  TestController::state_array state = TestController::state_array::Ones();
  for (double i = 0; i < controller->getNumTimesteps() - 1; i += 0.25)
  {
    TestController::state_array interpolated_state = controller->interpolateState(s_traj, i * controller->getDt());
    TestController::control_array result =
        controller->getCurrentControl(state, i * controller->getDt(), interpolated_state, traj, fb_state);
    EXPECT_FLOAT_EQ(result(0), i * 2);
  }
}

TEST_F(ControllerTests, saveControlHistoryHelper_1)
{
  int steps = 1;
  TestController::control_trajectory u =
      TestController::control_trajectory::Random(MockDynamics::CONTROL_DIM, NUM_TIMESTEPS);
  Eigen::Matrix<float, MockDynamics::CONTROL_DIM, 2> u_history;
  u_history.setOnes();

  controller->saveControlHistoryHelper(steps, u, u_history);

  for (int i = 0; i < MockDynamics::CONTROL_DIM; ++i)
  {
    EXPECT_FLOAT_EQ(u_history(i, 0), 1.0f) << "History column 0 failed";
    EXPECT_FLOAT_EQ(u_history(i, 1), u(i, steps - 1)) << "History column 1 failed";
  }
}

TEST_F(ControllerTests, saveControlHistoryHelper_2)
{
  int steps = 4;
  TestController::control_trajectory u =
      TestController::control_trajectory::Random(MockDynamics::CONTROL_DIM, NUM_TIMESTEPS);
  Eigen::Matrix<float, MockDynamics::CONTROL_DIM, 2> u_history;
  u_history.setOnes();

  controller->saveControlHistoryHelper(steps, u, u_history);

  for (int i = 0; i < MockDynamics::CONTROL_DIM; ++i)
  {
    EXPECT_FLOAT_EQ(u_history(i, 0), u(i, steps - 2)) << "History column 0 failed";
    EXPECT_FLOAT_EQ(u_history(i, 1), u(i, steps - 1)) << "History column 1 failed";
  }
}
