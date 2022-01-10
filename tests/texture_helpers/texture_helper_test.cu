#include <gtest/gtest.h>

#include <mppi/utils/test_helper.h>
#include <random>
#include <algorithm>

#include <mppi/utils/texture_helpers/texture_helper.cuh>

class TextureHelperTest : public testing::Test
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

class TextureHelperImpl : public TextureHelper<TextureHelperImpl, float4>
{
public:
  TextureHelperImpl(int number, cudaStream_t stream = 0) : TextureHelper<TextureHelperImpl, float4>(number, stream)
  {
  }

  ~TextureHelperImpl()
  {
    for (TextureParams<float4>& params : textures_)
    {
      params.allocated = false;
    }
  }

  std::vector<TextureParams<float4>> getTextures()
  {
    return textures_;
  }

  void setTestExtent(int index, cudaExtent extent)
  {
    this->textures_[index].extent = extent;
  }

  void setGpuMemStatus(bool flag)
  {
    this->GPUMemStatus_ = flag;
  }

  void setTestFlagsInParam(int index, bool update_mem, bool update_data, bool allocated)
  {
    this->textures_[index].update_mem = update_mem;
    this->textures_[index].update_data = update_data;
    this->textures_[index].allocated = allocated;
  }

  void copyDataToGPU(int index) override
  {
    copyDataToGPUCalled++;
    this->textures_[index].update_data = false;
  }

  void allocateCudaTexture(int index) override
  {
    allocateCudaTextureCalled++;
  }

  void createCudaTexture(int index) override
  {
    createCudaTextureCalled++;
    this->textures_[index].allocated = true;
    this->textures_[index].update_mem = false;
  }

  void clearCounters()
  {
    copyDataToGPUCalled = 0;
    allocateCudaTextureCalled = 0;
    createCudaTextureCalled = 0;
  }

  TextureParams<float4>* getTextureD()
  {
    return this->textures_d_;
  }

  TextureParams<float4>* getTextureDataPtr()
  {
    return this->textures_.data();
  }

  int copyDataToGPUCalled = 0;
  int allocateCudaTextureCalled = 0;
  int createCudaTextureCalled = 0;

  float4 queryTexture(const int index, const float3& input)
  {
    float4 result;
    result.x = input.x;
    result.y = input.y;
    result.z = input.z;
    result.w = index;
    return result;
  }
};

TEST_F(TextureHelperTest, Constructor)
{
  int number = 5;
  TextureHelperImpl helper = TextureHelperImpl(number);

  std::vector<TextureParams<float4>> textures = helper.getTextures();

  EXPECT_EQ(textures.size(), number);
  EXPECT_NE(helper.getTextureD(), nullptr);
  EXPECT_NE(helper.getTextureDataPtr(), nullptr);
  EXPECT_EQ(helper.getTextureD(), helper.getTextureDataPtr());

  for (const TextureParams<float4> texture : textures)
  {
    EXPECT_EQ(texture.use, false);
    EXPECT_EQ(texture.allocated, false);
    EXPECT_EQ(texture.update_data, false);
    EXPECT_EQ(texture.update_mem, false);

    EXPECT_EQ(texture.array_d, nullptr);
    EXPECT_EQ(texture.tex_d, 0);

    EXPECT_EQ(texture.resDesc.resType, cudaResourceTypeArray);
    EXPECT_EQ(texture.channelDesc.x, 32);
    EXPECT_EQ(texture.channelDesc.y, 32);
    EXPECT_EQ(texture.channelDesc.z, 32);
    EXPECT_EQ(texture.channelDesc.w, 32);

    EXPECT_EQ(texture.texDesc.addressMode[0], cudaAddressModeClamp);
    EXPECT_EQ(texture.texDesc.addressMode[1], cudaAddressModeClamp);
    EXPECT_EQ(texture.texDesc.filterMode, cudaFilterModeLinear);
    EXPECT_EQ(texture.texDesc.readMode, cudaReadModeElementType);
    EXPECT_EQ(texture.texDesc.normalizedCoords, 1);
  }
}

TEST_F(TextureHelperTest, UpdateOriginTest)
{
  int number = 5;
  TextureHelperImpl helper = TextureHelperImpl(number);
  helper.updateOrigin(0, make_float3(1, 2, 3));
  helper.updateOrigin(1, make_float3(4, 5, 6));

  std::vector<TextureParams<float4>> textures = helper.getTextures();
  EXPECT_FLOAT_EQ(textures[0].origin.x, 1);
  EXPECT_FLOAT_EQ(textures[0].origin.y, 2);
  EXPECT_FLOAT_EQ(textures[0].origin.z, 3);

  EXPECT_FLOAT_EQ(textures[1].origin.x, 4);
  EXPECT_FLOAT_EQ(textures[1].origin.y, 5);
  EXPECT_FLOAT_EQ(textures[1].origin.z, 6);
}

TEST_F(TextureHelperTest, UpdateRotationTest)
{
  int number = 5;
  TextureHelperImpl helper = TextureHelperImpl(number);

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(1, 2, 3);
  new_rot_mat[1] = make_float3(4, 5, 6);
  new_rot_mat[2] = make_float3(7, 8, 9);
  helper.updateRotation(0, new_rot_mat);

  std::vector<TextureParams<float4>> textures = helper.getTextures();

  EXPECT_FLOAT_EQ(textures[0].rotations[0].x, 1);
  EXPECT_FLOAT_EQ(textures[0].rotations[0].y, 2);
  EXPECT_FLOAT_EQ(textures[0].rotations[0].z, 3);
  EXPECT_FLOAT_EQ(textures[0].rotations[1].x, 4);
  EXPECT_FLOAT_EQ(textures[0].rotations[1].y, 5);
  EXPECT_FLOAT_EQ(textures[0].rotations[1].z, 6);
  EXPECT_FLOAT_EQ(textures[0].rotations[2].x, 7);
  EXPECT_FLOAT_EQ(textures[0].rotations[2].y, 8);
  EXPECT_FLOAT_EQ(textures[0].rotations[2].z, 9);

  new_rot_mat[2].x = 22;
  helper.updateRotation(1, new_rot_mat);
  textures = helper.getTextures();

  EXPECT_FLOAT_EQ(textures[0].rotations[0].x, 1);
  EXPECT_FLOAT_EQ(textures[0].rotations[0].y, 2);
  EXPECT_FLOAT_EQ(textures[0].rotations[0].z, 3);
  EXPECT_FLOAT_EQ(textures[0].rotations[1].x, 4);
  EXPECT_FLOAT_EQ(textures[0].rotations[1].y, 5);
  EXPECT_FLOAT_EQ(textures[0].rotations[1].z, 6);
  EXPECT_FLOAT_EQ(textures[0].rotations[2].x, 7);
  EXPECT_FLOAT_EQ(textures[0].rotations[2].y, 8);
  EXPECT_FLOAT_EQ(textures[0].rotations[2].z, 9);

  EXPECT_FLOAT_EQ(textures[1].rotations[0].x, 1);
  EXPECT_FLOAT_EQ(textures[1].rotations[0].y, 2);
  EXPECT_FLOAT_EQ(textures[1].rotations[0].z, 3);
  EXPECT_FLOAT_EQ(textures[1].rotations[1].x, 4);
  EXPECT_FLOAT_EQ(textures[1].rotations[1].y, 5);
  EXPECT_FLOAT_EQ(textures[1].rotations[1].z, 6);
  EXPECT_FLOAT_EQ(textures[1].rotations[2].x, 22);
  EXPECT_FLOAT_EQ(textures[1].rotations[2].y, 8);
  EXPECT_FLOAT_EQ(textures[1].rotations[2].z, 9);
}

TEST_F(TextureHelperTest, WorldPoseToMapPoseTest)
{
  int number = 5;
  TextureHelperImpl helper = TextureHelperImpl(number);

  // set it to be zero
  std::array<float3, 3> new_rot_mat{};
  helper.updateRotation(0, new_rot_mat);
  helper.updateOrigin(0, make_float3(0, 0, 0));

  float3 input = make_float3(0.1, 0.2, 0.3);
  float3 output;
  helper.worldPoseToMapPose(0, input, output);

  EXPECT_FLOAT_EQ(input.x, 0.1);
  EXPECT_FLOAT_EQ(input.y, 0.2);
  EXPECT_FLOAT_EQ(input.z, 0.3);
  EXPECT_FLOAT_EQ(output.x, 0);
  EXPECT_FLOAT_EQ(output.y, 0);
  EXPECT_FLOAT_EQ(output.z, 0);

  // set rot mat non zero
  new_rot_mat[0] = make_float3(1, 2, 3);
  new_rot_mat[1] = make_float3(4, 5, 6);
  new_rot_mat[2] = make_float3(7, 8, 9);
  helper.updateRotation(0, new_rot_mat);

  helper.worldPoseToMapPose(0, input, output);
  EXPECT_FLOAT_EQ(input.x, 0.1);
  EXPECT_FLOAT_EQ(input.y, 0.2);
  EXPECT_FLOAT_EQ(input.z, 0.3);
  EXPECT_FLOAT_EQ(output.x, 0.1 + 0.2 * 2 + 0.3 * 3);
  EXPECT_FLOAT_EQ(output.y, 0.1 * 4 + 0.2 * 5 + 0.3 * 6);
  EXPECT_FLOAT_EQ(output.z, 0.1 * 7 + 0.2 * 8 + 0.3 * 9);

  // set origin non zero
  new_rot_mat[0] = make_float3(0, 0, 0);
  new_rot_mat[1] = make_float3(0, 0, 0);
  new_rot_mat[2] = make_float3(0, 0, 0);
  helper.updateRotation(0, new_rot_mat);
  helper.updateOrigin(0, make_float3(0.1, 0.2, 0.3));

  helper.worldPoseToMapPose(0, input, output);
  EXPECT_FLOAT_EQ(input.x, 0.1);
  EXPECT_FLOAT_EQ(input.y, 0.2);
  EXPECT_FLOAT_EQ(input.z, 0.3);
  EXPECT_FLOAT_EQ(output.x, 0);
  EXPECT_FLOAT_EQ(output.y, 0);
  EXPECT_FLOAT_EQ(output.z, 0);

  // set rot mat non zero
  new_rot_mat[0] = make_float3(1, 2, 3);
  new_rot_mat[1] = make_float3(4, 5, 6);
  new_rot_mat[2] = make_float3(7, 8, 9);
  helper.updateRotation(0, new_rot_mat);
  input = make_float3(0.2, 0.4, 0.6);

  helper.worldPoseToMapPose(0, input, output);
  EXPECT_FLOAT_EQ(input.x, 0.2);
  EXPECT_FLOAT_EQ(input.y, 0.4);
  EXPECT_FLOAT_EQ(input.z, 0.6);
  EXPECT_FLOAT_EQ(output.x, 0.1 + 0.2 * 2 + 0.3 * 3);
  EXPECT_FLOAT_EQ(output.y, 0.1 * 4 + 0.2 * 5 + 0.3 * 6);
  EXPECT_FLOAT_EQ(output.z, 0.1 * 7 + 0.2 * 8 + 0.3 * 9);
}

TEST_F(TextureHelperTest, SetExtentTest)
{
  int number = 5;
  TextureHelperImpl helper = TextureHelperImpl(number);
  cudaExtent extent = make_cudaExtent(4, 5, 6);
  helper.setTestExtent(1, extent);

  helper.setExtent(0, extent);
  helper.setExtent(1, extent);

  std::vector<TextureParams<float4>> textures = helper.getTextures();

  EXPECT_TRUE(textures[0].update_mem);
  EXPECT_EQ(textures[0].extent.width, 4);
  EXPECT_EQ(textures[0].extent.height, 5);
  EXPECT_EQ(textures[0].extent.depth, 6);

  EXPECT_FALSE(textures[1].update_mem);
  EXPECT_EQ(textures[1].extent.width, 4);
  EXPECT_EQ(textures[1].extent.height, 5);
  EXPECT_EQ(textures[1].extent.depth, 6);
}

TEST_F(TextureHelperTest, MapPoseToTexCoordTest)
{
  int number = 5;
  TextureHelperImpl helper = TextureHelperImpl(number);

  float3 input = make_float3(1.0, 2.0, 3.0);
  float3 output;

  helper.updateResolution(0, 10);
  cudaExtent extent = make_cudaExtent(4, 5, 6);
  helper.setExtent(0, extent);

  helper.mapPoseToTexCoord(0, input, output);
  EXPECT_FLOAT_EQ(input.x, 1.0);
  EXPECT_FLOAT_EQ(input.y, 2.0);
  EXPECT_FLOAT_EQ(input.z, 3.0);
  EXPECT_FLOAT_EQ(output.x, (0.5 + 0.1) / 4);
  EXPECT_FLOAT_EQ(output.y, (0.5 + 0.2) / 5);
  EXPECT_FLOAT_EQ(output.z, (0.5 + 0.3) / 6);
}

TEST_F(TextureHelperTest, WorldPoseToTexCoordTest)
{
  int number = 5;
  TextureHelperImpl helper = TextureHelperImpl(number);

  float3 input = make_float3(0.2, 0.4, 0.6);
  float3 output;

  helper.updateResolution(0, 10);
  cudaExtent extent = make_cudaExtent(4, 5, 6);
  helper.setExtent(0, extent);

  std::array<float3, 3> new_rot_mat{};
  new_rot_mat[0] = make_float3(1, 2, 3);
  new_rot_mat[1] = make_float3(4, 5, 6);
  new_rot_mat[2] = make_float3(7, 8, 9);
  helper.updateRotation(0, new_rot_mat);
  helper.updateOrigin(0, make_float3(0.1, 0.2, 0.3));

  helper.worldPoseToTexCoord(0, input, output);
  EXPECT_FLOAT_EQ(input.x, 0.2);
  EXPECT_FLOAT_EQ(input.y, 0.4);
  EXPECT_FLOAT_EQ(input.z, 0.6);
  EXPECT_FLOAT_EQ(output.x, ((0.1 * 1 + 0.2 * 2 + 0.3 * 3) / 10 + 0.5) / 4);
  EXPECT_FLOAT_EQ(output.y, ((0.1 * 4 + 0.2 * 5 + 0.3 * 6) / 10 + 0.5) / 5);
  EXPECT_FLOAT_EQ(output.z, ((0.1 * 7 + 0.2 * 8 + 0.3 * 9) / 10 + 0.5) / 6);
}

TEST_F(TextureHelperTest, CopyToDeviceTest)
{
  int number = 8;
  TextureHelperImpl helper = TextureHelperImpl(number);

  helper.setTestFlagsInParam(0, false, false, false);
  helper.setTestFlagsInParam(1, false, false, true);
  helper.setTestFlagsInParam(2, false, true, false);
  helper.setTestFlagsInParam(3, false, true, true);
  helper.setTestFlagsInParam(4, true, false, false);
  helper.setTestFlagsInParam(5, true, false, true);
  helper.setTestFlagsInParam(6, true, true, false);
  helper.setTestFlagsInParam(7, true, true, true);

  // nothing should have happened since GPUMemStatus=False
  helper.copyToDevice();
  std::vector<TextureParams<float4>> textures = helper.getTextures();

  EXPECT_FALSE(textures[0].update_mem);
  EXPECT_FALSE(textures[0].update_data);
  EXPECT_FALSE(textures[0].allocated);

  EXPECT_FALSE(textures[1].update_mem);
  EXPECT_FALSE(textures[1].update_data);
  EXPECT_TRUE(textures[1].allocated);

  EXPECT_FALSE(textures[2].update_mem);
  EXPECT_TRUE(textures[2].update_data);
  EXPECT_FALSE(textures[2].allocated);

  EXPECT_FALSE(textures[3].update_mem);
  EXPECT_TRUE(textures[3].update_data);
  EXPECT_TRUE(textures[3].allocated);

  EXPECT_TRUE(textures[4].update_mem);
  EXPECT_FALSE(textures[4].update_data);
  EXPECT_FALSE(textures[4].allocated);

  EXPECT_TRUE(textures[5].update_mem);
  EXPECT_FALSE(textures[5].update_data);
  EXPECT_TRUE(textures[5].allocated);

  EXPECT_TRUE(textures[6].update_mem);
  EXPECT_TRUE(textures[6].update_data);
  EXPECT_FALSE(textures[6].allocated);

  EXPECT_TRUE(textures[7].update_mem);
  EXPECT_TRUE(textures[7].update_data);
  EXPECT_TRUE(textures[7].allocated);

  EXPECT_EQ(helper.copyDataToGPUCalled, 0);
  EXPECT_EQ(helper.allocateCudaTextureCalled, 0);
  EXPECT_EQ(helper.createCudaTextureCalled, 0);

  helper.clearCounters();
  helper.setGpuMemStatus(true);
  helper.copyToDevice();
  textures = helper.getTextures();

  EXPECT_FALSE(textures[0].update_mem);
  EXPECT_FALSE(textures[0].update_data);
  EXPECT_FALSE(textures[0].allocated);

  EXPECT_FALSE(textures[1].update_mem);
  EXPECT_FALSE(textures[1].update_data);
  EXPECT_TRUE(textures[1].allocated);

  EXPECT_FALSE(textures[2].update_mem);
  EXPECT_TRUE(textures[2].update_data);
  EXPECT_FALSE(textures[2].allocated);

  EXPECT_FALSE(textures[3].update_mem);
  EXPECT_FALSE(textures[3].update_data);  // was TRUE
  EXPECT_TRUE(textures[3].allocated);

  EXPECT_FALSE(textures[4].update_mem);  // was TRUE
  EXPECT_FALSE(textures[4].update_data);
  EXPECT_TRUE(textures[4].allocated);  // was FALSE

  EXPECT_FALSE(textures[5].update_mem);  // was TRUE
  EXPECT_FALSE(textures[5].update_data);
  EXPECT_TRUE(textures[5].allocated);

  EXPECT_FALSE(textures[6].update_mem);   // was TRUE
  EXPECT_FALSE(textures[6].update_data);  // was TRUE
  EXPECT_TRUE(textures[6].allocated);

  EXPECT_FALSE(textures[7].update_mem);   // was TRUE
  EXPECT_FALSE(textures[7].update_data);  // was TRUE
  EXPECT_TRUE(textures[7].allocated);

  EXPECT_EQ(helper.copyDataToGPUCalled, 3);
  EXPECT_EQ(helper.allocateCudaTextureCalled, 4);
  EXPECT_EQ(helper.createCudaTextureCalled, 4);
}

TEST_F(TextureHelperTest, AddNewTextureTest)
{
  int number = 8;
  TextureHelperImpl helper = TextureHelperImpl(number);
  cudaExtent extent = make_cudaExtent(4, 5, 6);
  helper.addNewTexture(extent);

  std::vector<TextureParams<float4>> textures = helper.getTextures();
  EXPECT_EQ(textures.size(), number + 1);
  for (int i = 0; i < number + 1; i++)
  {
    auto texture = textures.at(i);

    EXPECT_EQ(texture.use, false);
    // since GPUMemStatus_ = false, this should always be false
    EXPECT_EQ(texture.allocated, false);
    EXPECT_EQ(texture.update_data, false);
    EXPECT_EQ(texture.update_mem, false);
    EXPECT_EQ(texture.resDesc.resType, cudaResourceTypeArray);
    EXPECT_EQ(texture.channelDesc.x, 32);
    EXPECT_EQ(texture.channelDesc.y, 32);
    EXPECT_EQ(texture.channelDesc.z, 32);
    EXPECT_EQ(texture.channelDesc.w, 32);

    EXPECT_EQ(texture.texDesc.addressMode[0], cudaAddressModeClamp);
    EXPECT_EQ(texture.texDesc.addressMode[1], cudaAddressModeClamp);
    EXPECT_EQ(texture.texDesc.filterMode, cudaFilterModeLinear);
    EXPECT_EQ(texture.texDesc.readMode, cudaReadModeElementType);
    EXPECT_EQ(texture.texDesc.normalizedCoords, 1);
  }
  EXPECT_EQ(textures[8].extent.width, 4);
  EXPECT_EQ(textures[8].extent.height, 5);
  EXPECT_EQ(textures[8].extent.depth, 6);

  helper.setGpuMemStatus(true);
  helper.addNewTexture(extent);
  textures = helper.getTextures();
  EXPECT_EQ(textures.size(), number + 2);
  for (int i = 0; i < number + 2; i++)
  {
    auto texture = textures.at(i);

    EXPECT_EQ(texture.use, false);
    if (i == 9)
    {
      EXPECT_EQ(texture.allocated, true);
    }
    else
    {
      EXPECT_EQ(texture.allocated, false);
    }
    EXPECT_EQ(texture.update_data, false);
    EXPECT_EQ(texture.update_mem, false);
    EXPECT_EQ(texture.resDesc.resType, cudaResourceTypeArray);
    EXPECT_EQ(texture.channelDesc.x, 32);
    EXPECT_EQ(texture.channelDesc.y, 32);
    EXPECT_EQ(texture.channelDesc.z, 32);
    EXPECT_EQ(texture.channelDesc.w, 32);

    EXPECT_EQ(texture.texDesc.addressMode[0], cudaAddressModeClamp);
    EXPECT_EQ(texture.texDesc.addressMode[1], cudaAddressModeClamp);
    EXPECT_EQ(texture.texDesc.filterMode, cudaFilterModeLinear);
    EXPECT_EQ(texture.texDesc.readMode, cudaReadModeElementType);
    EXPECT_EQ(texture.texDesc.normalizedCoords, 1);
  }
  EXPECT_EQ(textures[9].extent.width, 4);
  EXPECT_EQ(textures[9].extent.height, 5);
  EXPECT_EQ(textures[9].extent.depth, 6);
}

// TEST_F(TextureHelperTest, queryTextureAtWorldPose) {
//  int number = 5;
//  TextureHelperImpl helper = TextureHelperImpl(number);
//
//  float3 input = make_float3(0.2,0.4,0.6);
//
//  helper.updateResolution(0, 10);
//  cudaExtent extent = make_cudaExtent(4,5,6);
//  helper.setExtent(0, extent);
//
//  std::array<float3, 3> new_rot_mat{};
//  new_rot_mat[0] = make_float3(1, 2, 3);
//  new_rot_mat[1] = make_float3(4, 5, 6);
//  new_rot_mat[2] = make_float3(7, 8, 9);
//  helper.updateRotation(0, new_rot_mat);
//  helper.updateOrigin(0, make_float3(0.1,0.2,0.3));
//
//  float4 output = helper.queryTextureAtWorldPose(0, input);
//  EXPECT_FLOAT_EQ(input.x, 0.2);
//  EXPECT_FLOAT_EQ(input.y, 0.4);
//  EXPECT_FLOAT_EQ(input.z, 0.6);
//  EXPECT_FLOAT_EQ(output.x, ((0.1*1+0.2*2+0.3*3)/10+0.5)/4);
//  EXPECT_FLOAT_EQ(output.y, ((0.1*4+0.2*5+0.3*6)/10+0.5)/5);
//  EXPECT_FLOAT_EQ(output.z, ((0.1*7+0.2*8+0.3*9)/10+0.5)/6);
//  EXPECT_FLOAT_EQ(output.w, 0);
//  // TODO
//}
