//
// Created by jason on 1/7/20.
//

#include <gtest/gtest.h>
#include <cost_functions/autorally/ar_standard_cost.cuh>
#include <cost_functions/ar_standard_cost_kernel_test.cuh>

TEST(ARStandardCost, Constructor) {
  ARStandardCost cost;
}

TEST(ARStandardCost, BindStream) {
  cudaStream_t stream;

  HANDLE_ERROR(cudaStreamCreate(&stream));

  ARStandardCost cost(stream);

  EXPECT_EQ(cost.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST(ARStandardCost, SetGetParamsHost) {
  ARStandardCost::ARStandardCostParams params;
  params.desired_speed = 25;
  params.num_timesteps = 100;
  params.r_c1.x = 0;
  params.r_c1.y = 1;
  params.r_c1.z = 2;
  ARStandardCost cost;

  cost.setParams(params);
  ARStandardCost::ARStandardCostParams result_params = cost.getParams();

  EXPECT_FLOAT_EQ(params.desired_speed, result_params.desired_speed);
  EXPECT_EQ(params.num_timesteps, result_params.num_timesteps);
  EXPECT_FLOAT_EQ(params.r_c1.x, result_params.r_c1.x);
  EXPECT_FLOAT_EQ(params.r_c1.y, result_params.r_c1.y);
  EXPECT_FLOAT_EQ(params.r_c1.z, result_params.r_c1.z);
}

TEST(ARStandardCost, GPUSetupAndParamsToDeviceTest) {
  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost;
  params.desired_speed = 25;
  params.num_timesteps = 100;
  params.r_c1.x = 0;
  params.r_c1.y = 1;
  params.r_c1.z = 2;
  cost.setParams(params);

  EXPECT_EQ(cost.GPUMemStatus_, false);

  cost.GPUSetup();

  EXPECT_EQ(cost.GPUMemStatus_, true);

  float desired_speed;
  int num_timesteps, height, width;
  float3 r_c1;
  launchParameterTestKernel(cost, desired_speed, num_timesteps, r_c1, width, height);

  EXPECT_FLOAT_EQ(desired_speed, 25);
  EXPECT_EQ(num_timesteps, 100);
  EXPECT_FLOAT_EQ(r_c1.x, 0);
  EXPECT_FLOAT_EQ(r_c1.y, 1);
  EXPECT_FLOAT_EQ(r_c1.z, 2);
  // neither should be set by this sequence
  EXPECT_EQ(width, -1);
  EXPECT_EQ(height, -1);

  params.desired_speed = 5;
  params.num_timesteps = 50;
  params.r_c1.x = 4;
  params.r_c1.y = 5;
  params.r_c1.z = 6;
  cost.setParams(params);
  cost.paramsToDevice();

  launchParameterTestKernel(cost, desired_speed, num_timesteps, r_c1, height, width);

  EXPECT_FLOAT_EQ(desired_speed, 5);
  EXPECT_EQ(num_timesteps, 50);
  EXPECT_FLOAT_EQ(r_c1.x, 4);
  EXPECT_FLOAT_EQ(r_c1.y, 5);
  EXPECT_FLOAT_EQ(r_c1.z, 6);

  // neither should be set by this sequence
  EXPECT_EQ(width, -1);
  EXPECT_EQ(height, -1);
}

TEST(ARStandardCost, changeCostmapSizeTestValidInputs) {
  ARStandardCost cost;
  cost.changeCostmapSize(4, 8);

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);
  EXPECT_EQ(cost.getTrackCostCPU().capacity(), 4*8);

  std::vector<float4> result;

  //launchCheckCudaArray(result, cost.getCudaArray(), 4*8);

  //EXPECT_EQ(result.size(), 4*8);
  // TODO verify that cuda is properly allocating the memory
  /*
  for(int i = 0; i < 4*8; i++) {
    EXPECT_FLOAT_EQ(result[i].x, 0);
    EXPECT_FLOAT_EQ(result[i].y, 0);
    EXPECT_FLOAT_EQ(result[i].z, 0);
    EXPECT_FLOAT_EQ(result[i].w, 0);
  }
   */
}

TEST(ARStandardCost, changeCostmapSizeTestFail) {
  ARStandardCost cost;
  cost.changeCostmapSize(4, 8);

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);
  EXPECT_EQ(cost.getTrackCostCPU().capacity(), 4*8);

  cost.changeCostmapSize(-1, -1);

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);
  EXPECT_EQ(cost.getTrackCostCPU().capacity(), 4*8);
}

TEST(ARStandardCost, clearCostmapTest) {
  ARStandardCost cost;
  cost.clearCostmapCPU(4, 8);

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);
  EXPECT_EQ(cost.getTrackCostCPU().capacity(), 4*8);

  for(int i = 0; i < 4 * 8; i++) {
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).x, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).y, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).z, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).w, 0);
  }
}

TEST(ARStandardCost, clearCostmapTestDefault) {
  ARStandardCost cost;
  cost.clearCostmapCPU(4, 8);

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);
  EXPECT_EQ(cost.getTrackCostCPU().capacity(), 4*8);

  for(int i = 0; i < 4 * 8; i++) {
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).x, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).y, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).z, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).w, 0);
  }

  cost.clearCostmapCPU();

  EXPECT_EQ(cost.getWidth(), 4);
  EXPECT_EQ(cost.getHeight(), 8);
  EXPECT_EQ(cost.getTrackCostCPU().capacity(), 4*8);

  for(int i = 0; i < 4 * 8; i++) {
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).x, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).y, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).z, 0);
    EXPECT_FLOAT_EQ(cost.getTrackCostCPU().at(i).w, 0);
  }
}

TEST(ARStandardCost, clearCostmapTestDefaultFail) {
  ARStandardCost cost;
  cost.clearCostmapCPU();

  EXPECT_EQ(cost.getWidth(), -1);
  EXPECT_EQ(cost.getHeight(), -1);
  EXPECT_EQ(cost.getTrackCostCPU().capacity(), 0);
}


TEST(ARStandardCost, LoadTrackDataTest) {
  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost;
  // TODO set parameters for cost map
  cost.setParams(params);
  cost.GPUSetup();
  float desired_speed;
  int num_timesteps, height, width;
  float3 r_c1;
  launchParameterTestKernel(cost, desired_speed, num_timesteps, r_c1, width, height);

  // TODO make path less stupid
  std::string test_location = "/home/jason/Documents/research/MPPI-Generic/resource/autorally/test/test_map.npz";
  Eigen::Matrix3f R;
  Eigen::Array3f trs;

  // load
  std::vector<float4> costmap = cost.loadTrackData(test_location, R, trs);

  EXPECT_FLOAT_EQ(costmap.at(0).x, 0);
  EXPECT_FLOAT_EQ(costmap.at(0).y, 0);
  EXPECT_FLOAT_EQ(costmap.at(0).z, 0);
  EXPECT_FLOAT_EQ(costmap.at(0).w, 0);
  EXPECT_FLOAT_EQ(costmap.at(1).x, 1);
  EXPECT_FLOAT_EQ(costmap.at(1).y, 10);
  EXPECT_FLOAT_EQ(costmap.at(1).z, 100);
  EXPECT_FLOAT_EQ(costmap.at(1).w, 1000);

  // check transformation, should not have a rotation
  EXPECT_FLOAT_EQ(R(0,0), 1.0 / (10));
  EXPECT_FLOAT_EQ(R(1,1), 1.0 / (20));
  EXPECT_FLOAT_EQ(R(2,2), 1.0);

  EXPECT_FLOAT_EQ(R(0, 1), 0);
  EXPECT_FLOAT_EQ(R(0, 2), 0);
  EXPECT_FLOAT_EQ(R(1, 0), 0);
  EXPECT_FLOAT_EQ(R(1, 2), 0);
  EXPECT_FLOAT_EQ(R(2, 0), 0);
  EXPECT_FLOAT_EQ(R(2, 1), 0);

  EXPECT_FLOAT_EQ(trs(0), 0.5);
  EXPECT_FLOAT_EQ(trs(1), 0.5);
  EXPECT_FLOAT_EQ(trs(2), 1);

  for(int i = 0; i < 2 * 10; i++) {
    for(int j = 0; j < 2 * 20; j++) {
      EXPECT_FLOAT_EQ(costmap.at(i*2*20 + j).x, i*2*20 + j);
      EXPECT_FLOAT_EQ(costmap.at(i*2*20 + j).y, (i*2*20 + j) * 10);
      EXPECT_FLOAT_EQ(costmap.at(i*2*20 + j).z, (i*2*20 + j) * 100);
      EXPECT_FLOAT_EQ(costmap.at(i*2*20 + j).w, (i*2*20 + j) * 1000);
    }
  }

}

TEST(ARStandardCost, costmapToTextureNoSizeTest) {
  ARStandardCost cost;
  cost.GPUSetup();

  cost.costmapToTexture();
}

TEST(ARStandardCost, costmapToTextureNoLoadTest) {
  ARStandardCost cost;
  cost.GPUSetup();

  cost.changeCostmapSize(4, 8);

  cost.costmapToTexture();

  std::vector<float4> results;
  std::vector<float2> query_points;
  float2 point;
  point.x = 0.0f;
  point.y = 0.0f;
  query_points.push_back(point);
  point.x = 1.0f;
  point.y = 0.0f;
  query_points.push_back(point);

  launchTextureTestKernel(cost, results, query_points);

  EXPECT_EQ(query_points.size(), results.size());

  EXPECT_FLOAT_EQ(results.at(0).x, 0);
  EXPECT_FLOAT_EQ(results.at(0).y, 0);
  EXPECT_FLOAT_EQ(results.at(0).z, 0);
  EXPECT_FLOAT_EQ(results.at(0).w, 0);

  EXPECT_FLOAT_EQ(results.at(1).x, 0);
  EXPECT_FLOAT_EQ(results.at(1).y, 0);
  EXPECT_FLOAT_EQ(results.at(1).z, 0);
  EXPECT_FLOAT_EQ(results.at(1).w, 0);
}

TEST(ARStandardCost, costmapToTextureLoadTest) {
  std::cout << "\n\n" << std::endl;
  ARStandardCost cost;
  cost.GPUSetup();

  std::string test_location = "/home/jason/Documents/research/MPPI-Generic/resource/autorally/test/test_map.npz";
  Eigen::Matrix3f R;
  Eigen::Array3f trs;

  std::vector<float4> costmap = cost.loadTrackData(test_location, R, trs);
  cost.costmapToTexture();

  std::vector<float4> results;
  std::vector<float2> query_points;
  float2 point;
  point.x = 0;
  point.y = 0;
  query_points.push_back(point);
  point.x = 0.1; // index 1 normalized
  point.y = 0;
  query_points.push_back(point);

  launchTextureTestKernel(cost, results, query_points);

  EXPECT_EQ(query_points.size(), results.size());

  EXPECT_FLOAT_EQ(results.at(0).x, 0);
  EXPECT_FLOAT_EQ(results.at(0).y, 0);
  EXPECT_FLOAT_EQ(results.at(0).z, 0);
  EXPECT_FLOAT_EQ(results.at(0).w, 0);
  EXPECT_FLOAT_EQ(results.at(1).x, 1);
  EXPECT_FLOAT_EQ(results.at(1).y, 10);
  EXPECT_FLOAT_EQ(results.at(1).z, 100);
  EXPECT_FLOAT_EQ(results.at(1).w, 1000);
}
