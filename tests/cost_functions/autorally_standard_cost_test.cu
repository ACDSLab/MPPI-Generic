//
// Created by jason on 1/7/20.
//

#include <gtest/gtest.h>
#include <cost_functions/autorally/ar_standard_cost.cuh>
#include <cost_functions/ar_standard_cost_kernel_test.cuh>

// Auto-generated header file
#include <autorally_test_map.h>

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
  params.speed_coeff = 2;
  params.track_coeff = 100;
  params.max_slip_ang = 1.5;
  params.slip_penalty = 1000;
  params.track_slop = 0.2;
  params.crash_coeff = 10000;
  params.steering_coeff = 20;
  params.throttle_coeff = 10;
  params.boundary_threshold = 10;
  params.discount = 0.9;
  params.grid_res = 2;
  params.num_timesteps = 100;

  params.r_c1.x = 0;
  params.r_c1.y = 1;
  params.r_c1.z = 2;
  params.r_c2.x = 3;
  params.r_c2.y = 4;
  params.r_c2.z = 5;
  params.trs.x = 6;
  params.trs.y = 7;
  params.trs.z = 8;
  ARStandardCost cost;

  cost.setParams(params);
  ARStandardCost::ARStandardCostParams result_params = cost.getParams();

  EXPECT_FLOAT_EQ(params.desired_speed, result_params.desired_speed);
  EXPECT_FLOAT_EQ(params.speed_coeff, result_params.speed_coeff);
  EXPECT_FLOAT_EQ(params.track_coeff, result_params.track_coeff);
  EXPECT_FLOAT_EQ(params.max_slip_ang, result_params.max_slip_ang);
  EXPECT_FLOAT_EQ(params.slip_penalty, result_params.slip_penalty);
  EXPECT_FLOAT_EQ(params.track_slop, result_params.track_slop);
  EXPECT_FLOAT_EQ(params.crash_coeff, result_params.crash_coeff);
  EXPECT_FLOAT_EQ(params.steering_coeff, result_params.steering_coeff);
  EXPECT_FLOAT_EQ(params.throttle_coeff, result_params.throttle_coeff);
  EXPECT_FLOAT_EQ(params.boundary_threshold, result_params.boundary_threshold);
  EXPECT_FLOAT_EQ(params.discount, result_params.discount);
  EXPECT_EQ(params.grid_res, result_params.grid_res);
  EXPECT_EQ(params.num_timesteps, result_params.num_timesteps);
  EXPECT_FLOAT_EQ(params.r_c1.x, result_params.r_c1.x);
  EXPECT_FLOAT_EQ(params.r_c1.y, result_params.r_c1.y);
  EXPECT_FLOAT_EQ(params.r_c1.z, result_params.r_c1.z);
  EXPECT_FLOAT_EQ(params.r_c2.x, result_params.r_c2.x);
  EXPECT_FLOAT_EQ(params.r_c2.y, result_params.r_c2.y);
  EXPECT_FLOAT_EQ(params.r_c2.z, result_params.r_c2.z);
  EXPECT_FLOAT_EQ(params.trs.x, result_params.trs.x);
  EXPECT_FLOAT_EQ(params.trs.y, result_params.trs.y);
  EXPECT_FLOAT_EQ(params.trs.z, result_params.trs.z);
}

TEST(ARStandardCost, GPUSetupAndParamsToDeviceTest) {

  // TODO make sre GPUMemstatus is false on the GPU so deallocation can be automatic
  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost;
  params.desired_speed = 25;
  params.speed_coeff = 2;
  params.track_coeff = 100;
  params.max_slip_ang = 1.5;
  params.slip_penalty = 1000;
  params.track_slop = 0.2;
  params.crash_coeff = 10000;
  params.steering_coeff = 20;
  params.throttle_coeff = 10;
  params.boundary_threshold = 10;
  params.discount = 0.9;
  params.grid_res = 2;
  params.num_timesteps = 100;

  params.r_c1.x = 0;
  params.r_c1.y = 1;
  params.r_c1.z = 2;
  params.r_c2.x = 3;
  params.r_c2.y = 4;
  params.r_c2.z = 5;
  params.trs.x = 6;
  params.trs.y = 7;
  params.trs.z = 8;
  cost.setParams(params);

  EXPECT_EQ(cost.GPUMemStatus_, false);

  cost.GPUSetup();

  EXPECT_EQ(cost.GPUMemStatus_, true);


  ARStandardCost::ARStandardCostParams result_params;
  int width, height;
  launchParameterTestKernel(cost, result_params, width, height);

  EXPECT_FLOAT_EQ(params.desired_speed, result_params.desired_speed);
  EXPECT_FLOAT_EQ(params.speed_coeff, result_params.speed_coeff);
  EXPECT_FLOAT_EQ(params.track_coeff, result_params.track_coeff);
  EXPECT_FLOAT_EQ(params.max_slip_ang, result_params.max_slip_ang);
  EXPECT_FLOAT_EQ(params.slip_penalty, result_params.slip_penalty);
  EXPECT_FLOAT_EQ(params.track_slop, result_params.track_slop);
  EXPECT_FLOAT_EQ(params.crash_coeff, result_params.crash_coeff);
  EXPECT_FLOAT_EQ(params.steering_coeff, result_params.steering_coeff);
  EXPECT_FLOAT_EQ(params.throttle_coeff, result_params.throttle_coeff);
  EXPECT_FLOAT_EQ(params.boundary_threshold, result_params.boundary_threshold);
  EXPECT_FLOAT_EQ(params.discount, result_params.discount);
  EXPECT_EQ(params.grid_res, result_params.grid_res);
  EXPECT_EQ(params.num_timesteps, result_params.num_timesteps);
  EXPECT_FLOAT_EQ(params.r_c1.x, result_params.r_c1.x);
  EXPECT_FLOAT_EQ(params.r_c1.y, result_params.r_c1.y);
  EXPECT_FLOAT_EQ(params.r_c1.z, result_params.r_c1.z);
  EXPECT_FLOAT_EQ(params.r_c2.x, result_params.r_c2.x);
  EXPECT_FLOAT_EQ(params.r_c2.y, result_params.r_c2.y);
  EXPECT_FLOAT_EQ(params.r_c2.z, result_params.r_c2.z);
  EXPECT_FLOAT_EQ(params.trs.x, result_params.trs.x);
  EXPECT_FLOAT_EQ(params.trs.y, result_params.trs.y);
  EXPECT_FLOAT_EQ(params.trs.z, result_params.trs.z);
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

  launchParameterTestKernel(cost, result_params, width, height);

  EXPECT_FLOAT_EQ(params.desired_speed, result_params.desired_speed);
  EXPECT_FLOAT_EQ(params.speed_coeff, result_params.speed_coeff);
  EXPECT_FLOAT_EQ(params.track_coeff, result_params.track_coeff);
  EXPECT_FLOAT_EQ(params.max_slip_ang, result_params.max_slip_ang);
  EXPECT_FLOAT_EQ(params.slip_penalty, result_params.slip_penalty);
  EXPECT_FLOAT_EQ(params.track_slop, result_params.track_slop);
  EXPECT_FLOAT_EQ(params.crash_coeff, result_params.crash_coeff);
  EXPECT_FLOAT_EQ(params.steering_coeff, result_params.steering_coeff);
  EXPECT_FLOAT_EQ(params.throttle_coeff, result_params.throttle_coeff);
  EXPECT_FLOAT_EQ(params.boundary_threshold, result_params.boundary_threshold);
  EXPECT_FLOAT_EQ(params.discount, result_params.discount);
  EXPECT_EQ(params.grid_res, result_params.grid_res);
  EXPECT_EQ(params.num_timesteps, result_params.num_timesteps);
  EXPECT_FLOAT_EQ(params.r_c1.x, result_params.r_c1.x);
  EXPECT_FLOAT_EQ(params.r_c1.y, result_params.r_c1.y);
  EXPECT_FLOAT_EQ(params.r_c1.z, result_params.r_c1.z);
  EXPECT_FLOAT_EQ(params.r_c2.x, result_params.r_c2.x);
  EXPECT_FLOAT_EQ(params.r_c2.y, result_params.r_c2.y);
  EXPECT_FLOAT_EQ(params.r_c2.z, result_params.r_c2.z);
  EXPECT_FLOAT_EQ(params.trs.x, result_params.trs.x);
  EXPECT_FLOAT_EQ(params.trs.y, result_params.trs.y);
  EXPECT_FLOAT_EQ(params.trs.z, result_params.trs.z);

  // neither should be set by this sequence
  EXPECT_EQ(width, -1);
  EXPECT_EQ(height, -1);
}

TEST(ARStandardCost, coorTransformTest) {
  float x,y,u,v,w;

  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost;

  x = 0;
  y = 10;

  params.r_c1.x = 0;
  params.r_c1.y = 1;
  params.r_c1.z = 2;
  params.r_c2.x = 3;
  params.r_c2.y = 4;
  params.r_c2.z = 5;
  params.trs.x = 6;
  params.trs.y = 7;
  params.trs.z = 8;
  cost.setParams(params);

  cost.coorTransform(x, y, &u, &v, &w);

  EXPECT_FLOAT_EQ(u, 36);
  EXPECT_FLOAT_EQ(v, 47);
  EXPECT_FLOAT_EQ(w, 58);
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

TEST(ARStandardCost, updateTransformCPUTest) {
  ARStandardCost cost;
  Eigen::MatrixXf R(3, 3);
  Eigen::ArrayXf trs(3);
  R(0,0) = 1;
  R(0,1) = 2;
  R(0,2) = 3;

  R(1,0) = 4;
  R(1,1) = 5;
  R(1,2) = 6;

  R(2,0) = 7;
  R(2,1) = 8;
  R(2,2) = 9;

  trs(0) = 10;
  trs(1) = 11;
  trs(2) = 12;

  cost.updateTransform(R, trs);

  /*
   * Prospective transform matrix
   * r_c1.x, r_c2.x, trs.x
   * r_c1.y, r_c2.y, trs.y
   * r_c1.z, r_c2.z, trs.z
   */

  EXPECT_FLOAT_EQ(cost.getParams().r_c1.x, 1);
  EXPECT_FLOAT_EQ(cost.getParams().r_c2.x, 2);
  EXPECT_FLOAT_EQ(cost.getParams().trs.x, 10);

  EXPECT_FLOAT_EQ(cost.getParams().r_c1.y, 4);
  EXPECT_FLOAT_EQ(cost.getParams().r_c2.y, 5);
  EXPECT_FLOAT_EQ(cost.getParams().trs.y, 11);

  EXPECT_FLOAT_EQ(cost.getParams().r_c1.z, 7);
  EXPECT_FLOAT_EQ(cost.getParams().r_c2.z, 8);
  EXPECT_FLOAT_EQ(cost.getParams().trs.z, 12);

  EXPECT_EQ(cost.GPUMemStatus_, false);
}

TEST(ARStandardCost, updateTransformGPUTest) {

  ARStandardCost cost;
  cost.GPUSetup();
  Eigen::MatrixXf R(3, 3);
  Eigen::ArrayXf trs(3);
  R(0,0) = 1;
  R(0,1) = 2;
  R(0,2) = 3;

  R(1,0) = 4;
  R(1,1) = 5;
  R(1,2) = 6;

  R(2,0) = 7;
  R(2,1) = 8;
  R(2,2) = 9;

  trs(0) = 10;
  trs(1) = 11;
  trs(2) = 12;

  cost.updateTransform(R, trs);

  /*
   * Prospective transform matrix
   * r_c1.x, r_c2.x, trs.x
   * r_c1.y, r_c2.y, trs.y
   * r_c1.z, r_c2.z, trs.z
   */

  std::vector<float3> results;
  launchTransformTestKernel(results, cost);

  EXPECT_EQ(results.size(), 3);

  EXPECT_FLOAT_EQ(results.at(0).x, 1);
  EXPECT_FLOAT_EQ(results.at(0).y, 4);
  EXPECT_FLOAT_EQ(results.at(0).z, 7);

  EXPECT_FLOAT_EQ(results.at(1).x, 2);
  EXPECT_FLOAT_EQ(results.at(1).y, 5);
  EXPECT_FLOAT_EQ(results.at(1).z, 8);

  EXPECT_FLOAT_EQ(results.at(2).x, 10);
  EXPECT_FLOAT_EQ(results.at(2).y, 11);
  EXPECT_FLOAT_EQ(results.at(2).z, 12);

  EXPECT_EQ(cost.GPUMemStatus_, true);
}


TEST(ARStandardCost, LoadTrackDataTest) {
  ARStandardCost::ARStandardCostParams params;
  ARStandardCost cost;
  // TODO set parameters for cost map
  cost.setParams(params);
  cost.GPUSetup();

  Eigen::Matrix3f R;
  Eigen::Array3f trs;

  // load
  std::vector<float4> costmap = cost.loadTrackData(mppi::tests::test_map_file, R, trs);

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

  Eigen::Matrix3f R;
  Eigen::Array3f trs;

  std::vector<float4> costmap = cost.loadTrackData(mppi::tests::test_map_file, R, trs);
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

