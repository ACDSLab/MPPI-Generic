#include <gtest/gtest.h>

#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>

#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>
#include <mppi/utils/texture_test_kernels.cuh>

class TwoDTextureHelperTest : public testing::Test
{
protected:
  void SetUp() override
  {
    generator = std::default_random_engine(7.0);
    distribution = std::normal_distribution<float>(100.0, 2.0);
  }

  void TearDown() override
  {
  }

  std::default_random_engine generator;
  std::normal_distribution<float> distribution;
};

TEST_F(TwoDTextureHelperTest, Constructor)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(4);

  EXPECT_EQ(helper.getCpuValues().size(), 4);
}

TEST_F(TwoDTextureHelperTest, TwoDAllocateCudaTextureTest)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(2);

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);
  extent = make_cudaExtent(30, 40, 0);
  helper.setExtent(1, extent);

  helper.GPUSetup();

  std::vector<TextureParams<float4>> textures = helper.getTextures();
  EXPECT_EQ(textures[0].array_d, nullptr);
  EXPECT_EQ(textures[0].tex_d, 0);
  EXPECT_TRUE(textures[0].update_mem);
  EXPECT_FALSE(textures[0].allocated);

  helper.allocateCudaTexture(0);
  textures = helper.getTextures();
  EXPECT_NE(textures[0].array_d, nullptr);
  EXPECT_EQ(textures[0].tex_d, 0);
  EXPECT_TRUE(textures[0].update_mem);
  EXPECT_FALSE(textures[0].allocated);
}

TEST_F(TwoDTextureHelperTest, TwoDAllocateAndCreateCudaTexture)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(2);

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);

  helper.GPUSetup();

  std::vector<TextureParams<float4>> textures = helper.getTextures();
  EXPECT_EQ(textures[0].array_d, nullptr);
  EXPECT_EQ(textures[0].tex_d, 0);
  EXPECT_TRUE(textures[0].update_mem);
  EXPECT_FALSE(textures[0].allocated);

  helper.allocateCudaTexture(0);
  helper.createCudaTexture(0);
  textures = helper.getTextures();
  EXPECT_NE(textures[0].array_d, nullptr);
  EXPECT_NE(textures[0].tex_d, 0);
  EXPECT_FALSE(textures[0].update_mem);
  EXPECT_TRUE(textures[0].allocated);
}

TEST_F(TwoDTextureHelperTest, UpdateTexture)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(2);

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);
  extent = make_cudaExtent(30, 40, 0);
  helper.setExtent(1, extent);

  std::vector<float4> data_vec;
  data_vec.resize(30 * 40);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = make_float4(i, i + 1, i + 2, i + 3);
  }

  helper.updateTexture(1, data_vec);

  std::vector<TextureParams<float4>> textures = helper.getTextures();
  EXPECT_TRUE(textures[1].update_data);

  auto cpu_values = helper.getCpuValues()[1];
  for (int i = 0; i < data_vec.size(); i++)
  {
    EXPECT_FLOAT_EQ(cpu_values[i].x, data_vec[i].x);
    EXPECT_FLOAT_EQ(cpu_values[i].y, data_vec[i].y);
    EXPECT_FLOAT_EQ(cpu_values[i].z, data_vec[i].z);
    EXPECT_FLOAT_EQ(cpu_values[i].w, data_vec[i].w);
  }
}

TEST_F(TwoDTextureHelperTest, UpdateTextureColumnMajor)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(2);
  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);

  std::vector<float4> data_vec;
  data_vec.resize(10 * 20);
  // just in case I am stupid
  std::set<float> total_set;
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = make_float4(i, i + 1, i + 2, i + 3);
    total_set.insert(i);
  }

  helper.setColumnMajor(0, true);
  helper.updateTexture(0, data_vec);

  // returns a rowMajor vector
  auto cpu_values = helper.getCpuValues()[0];

  for (int i = 0; i < 20; i++)
  {
    for (int j = 0; j < 10; j++)
    {
      int columnMajorIndex = i * 10 + j;
      int rowVectorIndex = j * 20 + i;
      EXPECT_FLOAT_EQ(cpu_values[rowVectorIndex].x, columnMajorIndex) << " at index: " << i;
      EXPECT_EQ(total_set.erase(rowVectorIndex), 1);
    }
  }
  EXPECT_EQ(total_set.size(), 0);
}

TEST_F(TwoDTextureHelperTest, UpdateTextureException)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(2);

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);
  extent = make_cudaExtent(30, 40, 0);
  helper.setExtent(1, extent);

  std::vector<float4> data_vec;
  data_vec.resize(30 * 40);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = make_float4(i, i + 1, i + 2, i + 3);
  }

  EXPECT_THROW(helper.updateTexture(0, data_vec), std::runtime_error);
}

void checkEqual(float4 computed, float val)
{
  EXPECT_FLOAT_EQ(computed.x, val);
  EXPECT_FLOAT_EQ(computed.y, val + 1);
  EXPECT_FLOAT_EQ(computed.z, val + 2);
  EXPECT_FLOAT_EQ(computed.w, val + 3);
}

TEST_F(TwoDTextureHelperTest, UpdateTextureNewSize)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(2);

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);
  extent = make_cudaExtent(30, 40, 0);
  helper.setExtent(1, extent);

  std::vector<float4> data_vec;
  data_vec.resize(30 * 40);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = make_float4(i, i + 1, i + 2, i + 3);
  }

  helper.updateTexture(0, data_vec, extent);

  std::vector<TextureParams<float4>> textures = helper.getTextures();
  EXPECT_TRUE(textures[0].update_data);
  EXPECT_TRUE(textures[0].update_mem);

  auto cpu_values = helper.getCpuValues()[0];
  for (int i = 0; i < data_vec.size(); i++)
  {
    checkEqual(cpu_values[i], data_vec[i].x);
  }
}

TEST_F(TwoDTextureHelperTest, CopyDataToGPU)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(1);
  helper.GPUSetup();

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);

  std::vector<float4> data_vec;
  data_vec.resize(10 * 20);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = make_float4(i, i + 1, i + 2, i + 3);
  }

  helper.updateTexture(0, data_vec);
  helper.allocateCudaTexture(0);
  helper.createCudaTexture(0);
  helper.copyDataToGPU(0, true);

  std::vector<float4> query_points;
  query_points.push_back(make_float4(0.0, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.05, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.95, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(1.0, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.45, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.5, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.55, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.025, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.05, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.075, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.975, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 1.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.475, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.5, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.525, 0.0, 0.0));

  auto results = getTextureAtPointsKernel<TwoDTextureHelper<float4>, float4>(helper, query_points);

  EXPECT_FLOAT_EQ(results.size(), query_points.size());
  checkEqual(results[0], 0.0);
  checkEqual(results[1], 0.0);

  checkEqual(results[2], 9.0);
  checkEqual(results[3], 9.0);

  checkEqual(results[4], 4.0);
  checkEqual(results[5], 4.5);
  checkEqual(results[6], 5.0);

  checkEqual(results[7], 0.0);
  checkEqual(results[8], 0.0);
  checkEqual(results[9], 5.0);
  checkEqual(results[10], 10.0);

  checkEqual(results[11], 190.0);
  checkEqual(results[12], 190.0);

  checkEqual(results[13], 90.0);
  checkEqual(results[14], 95.0);
  checkEqual(results[15], 100.0);
}

TEST_F(TwoDTextureHelperTest, QueryTextureAtMapPose)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(1);
  helper.GPUSetup();

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);

  std::vector<float4> data_vec;
  data_vec.resize(10 * 20);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = make_float4(i, i + 1, i + 2, i + 3);
  }

  helper.updateTexture(0, data_vec);
  helper.updateResolution(0, 10);
  helper.allocateCudaTexture(0);
  helper.createCudaTexture(0);
  helper.copyParamsToGPU(0);
  helper.copyDataToGPU(0, true);

  std::vector<float4> query_points;
  query_points.push_back(make_float4(0.0, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.05, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.95, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(1.0, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.45, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.5, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.55, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.025, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.05, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.075, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.975, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 1.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.475, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.5, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.525, 0.0, 0.0));

  for (int i = 0; i < query_points.size(); i++)
  {
    // resolution * normalized handling
    query_points[i].x = query_points[i].x * 10 * 10;
    query_points[i].y = query_points[i].y * 10 * 20;
    query_points[i].z = query_points[i].z * 10;
  }

  auto results = getTextureAtMapPointsKernel<TwoDTextureHelper<float4>, float4>(helper, query_points);

  EXPECT_FLOAT_EQ(results.size(), query_points.size());
  checkEqual(results[0], 0.0);
  checkEqual(results[1], 0.0);

  checkEqual(results[2], 9.0);
  checkEqual(results[3], 9.0);

  checkEqual(results[4], 4.0);
  checkEqual(results[5], 4.5);
  checkEqual(results[6], 5.0);

  checkEqual(results[7], 0.0);
  checkEqual(results[8], 0.0);
  checkEqual(results[9], 5.0);
  checkEqual(results[10], 10.0);

  checkEqual(results[11], 190.0);
  checkEqual(results[12], 190.0);

  checkEqual(results[13], 90.0);
  checkEqual(results[14], 95.0);
  checkEqual(results[15], 100.0);
}

TEST_F(TwoDTextureHelperTest, QueryTextureAtWorldPose)
{
  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(1);
  helper.GPUSetup();

  cudaExtent extent = make_cudaExtent(10, 20, 0);
  helper.setExtent(0, extent);

  std::vector<float4> data_vec;
  data_vec.resize(10 * 20);
  for (int i = 0; i < data_vec.size(); i++)
  {
    data_vec[i] = make_float4(i, i + 1, i + 2, i + 3);
  }

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(0, 1, 0);
  new_rot_mat[1] = make_float3(1, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 1);
  helper.updateRotation(0, new_rot_mat);
  helper.updateOrigin(0, make_float3(1, 2, 3));

  helper.updateTexture(0, data_vec);
  helper.updateResolution(0, 10);
  helper.allocateCudaTexture(0);
  helper.createCudaTexture(0);
  helper.copyParamsToGPU(0);
  helper.copyDataToGPU(0, true);

  std::vector<float4> query_points;
  query_points.push_back(make_float4(0.0, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.05, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.95, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(1.0, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.45, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.5, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.55, 0.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.0, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.025, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.05, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.075, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.975, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 1.0, 0.0, 0.0));

  query_points.push_back(make_float4(0.0, 0.475, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.5, 0.0, 0.0));
  query_points.push_back(make_float4(0.0, 0.525, 0.0, 0.0));

  for (int i = 0; i < query_points.size(); i++)
  {
    if (i == 4)
    {
      printf("correct tex value at %d: %f %f\n", i, query_points[i].x, query_points[i].y);
    }
    float4 temp = query_points[i];
    // resolution * normalized handling
    query_points[i].x = temp.y * 10.0 * 20.0 + 1;
    query_points[i].y = temp.x * 10.0 * 10.0 + 2;
    query_points[i].z = temp.z * 10.0 + 3;
    if (i == 4)
    {
      printf("starting input at %d: %f %f\n", i, query_points[i].x, query_points[i].y);
    }
  }

  auto results = getTextureAtWorldPointsKernel<TwoDTextureHelper<float4>, float4>(helper, query_points);

  EXPECT_FLOAT_EQ(results.size(), query_points.size());
  checkEqual(results[0], 0.0);
  checkEqual(results[1], 0.0);

  checkEqual(results[2], 9.0);
  checkEqual(results[3], 9.0);

  checkEqual(results[4], 4.0);
  checkEqual(results[5], 4.5);
  checkEqual(results[6], 5.0);

  checkEqual(results[7], 0.0);
  checkEqual(results[8], 0.0);
  checkEqual(results[9], 5.0);
  checkEqual(results[10], 10.0);

  checkEqual(results[11], 190.0);
  checkEqual(results[12], 190.0);

  checkEqual(results[13], 90.0);
  checkEqual(results[14], 95.0);
  checkEqual(results[15], 100.0);
}

// TEST_F(TwoDTextureHelperTest, CopyToDeviceTest)
//{
//  int number = 8;
//  TwoDTextureHelper<float4> helper = TwoDTextureHelper<float4>(number);
//
//  //helper.setTestFlagsInParam(0, false, false, false);
//  //helper.setTestFlagsInParam(1, false, false, true);
//  //helper.setTestFlagsInParam(2, false, true, false);
//  //helper.setTestFlagsInParam(3, false, true, true);
//  //helper.setTestFlagsInParam(4, true, false, false);
//  //helper.setTestFlagsInParam(5, true, false, true);
//  //helper.setTestFlagsInParam(6, true, true, false);
//  //helper.setTestFlagsInParam(7, true, true, true);
//
//  // nothing should have happened since GPUMemStatus=False
//  helper.GPUSetup();
//  helper.copyToDevice();
//  std::vector<TextureParams<float4>> textures = helper.getTextures();
//  textures = helper.getTextures();
//
//  EXPECT_FALSE(textures[0].update_mem);
//  EXPECT_FALSE(textures[0].update_data);
//  EXPECT_FALSE(textures[0].allocated);
//
//  EXPECT_FALSE(textures[1].update_mem);
//  EXPECT_FALSE(textures[1].update_data);
//  EXPECT_TRUE(textures[1].allocated);
//
//  EXPECT_FALSE(textures[2].update_mem);
//  EXPECT_TRUE(textures[2].update_data);
//  EXPECT_FALSE(textures[2].allocated);
//
//  EXPECT_FALSE(textures[3].update_mem);
//  EXPECT_FALSE(textures[3].update_data);  // was TRUE
//  EXPECT_TRUE(textures[3].allocated);
//
//  EXPECT_FALSE(textures[4].update_mem);  // was TRUE
//  EXPECT_FALSE(textures[4].update_data);
//  EXPECT_TRUE(textures[4].allocated);  // was FALSE
//
//  EXPECT_FALSE(textures[5].update_mem);  // was TRUE
//  EXPECT_FALSE(textures[5].update_data);
//  EXPECT_TRUE(textures[5].allocated);
//
//  EXPECT_FALSE(textures[6].update_mem);   // was TRUE
//  EXPECT_FALSE(textures[6].update_data);  // was TRUE
//  EXPECT_TRUE(textures[6].allocated);
//
//  EXPECT_FALSE(textures[7].update_mem);   // was TRUE
//  EXPECT_FALSE(textures[7].update_data);  // was TRUE
//  EXPECT_TRUE(textures[7].allocated);
//
//  //EXPECT_EQ(helper.copyDataToGPUCalled, 3);
//  //EXPECT_EQ(helper.allocateCudaTextureCalled, 4);
//  //EXPECT_EQ(helper.createCudaTextureCalled, 4);
//}
