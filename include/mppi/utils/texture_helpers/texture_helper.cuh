//
// Created by jason on 1/5/22.
//

#ifndef MPPIGENERIC_TEXTURE_HELPER_CUH
#define MPPIGENERIC_TEXTURE_HELPER_CUH

#include <mppi/utils/managed.cuh>

template <class DATA_T>
struct TextureParams {
  cudaExtent extent;

  cudaArray* array_d = nullptr;
  cudaTextureObject_t tex_d = 0;
  cudaChannelFormatDesc channelDesc;
  cudaResourceDesc resDesc;
  cudaTextureDesc texDesc;

  float3 origin;
  float3 rotations[3];
  float resolution;

  bool column_major = false;
  bool use = false;
  bool allocated = false;
  bool update_data = false;
  bool update_mem = false; // indicates the GPU structure should be updated at the next convenient time
  bool update_params = false;

  TextureParams() {
    resDesc.resType = cudaResourceTypeArray;
    channelDesc = cudaCreateChannelDesc<DATA_T>();

    // clamp
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
  }
};

template<class TEX_T, class DATA_T>
class TextureHelper : public Managed {
protected:
  TextureHelper(int number, cudaStream_t stream = 0);

public:
  virtual ~TextureHelper();

  void GPUSetup();

  static void freeCudaMem(TextureParams<DATA_T>& texture);
  virtual void freeCudaMem();

  /**
   * helper method to deallocate the index before allocating new ones
   */
  virtual void allocateCudaTexture(int index);
  /**
   * helper method to create a cuda texture
   * @param index
   */
  virtual void createCudaTexture(int index, bool sync=true);

  /**
   * Copies texture information to the GPU version of the object
   */
  virtual void copyToDevice(bool synchronize=false);

  /**
   *
   */
  virtual void addNewTexture(const cudaExtent& extent);

  __host__ __device__ void worldPoseToMapPose(const int index, const float3& input, float3& output);
  __host__ __device__ void mapPoseToTexCoord(const int index, const float3& input, float3& output);
  __host__ __device__ void worldPoseToTexCoord(const int index, const float3& input, float3& output);
  __device__ DATA_T queryTextureAtWorldPose(const int index, const float3& input);
  __device__ DATA_T queryTextureAtMapPose(int index, const float3& input);

  virtual void updateOrigin(int index, float3 new_origin);
  virtual void updateRotation(int index, std::array<float3, 3>& new_rotation);
  virtual void updateResolution(int index, float resolution);
  virtual void setExtent(int index, cudaExtent& extent);
  virtual void copyDataToGPU(int index, bool sync=false) = 0;
  virtual void copyParamsToGPU(int index, bool sync=false);
  virtual void setColumnMajor(int index, bool val) {
    this->textures_[index].column_major = val;
  }

  std::vector<TextureParams<float4>> getTextures()
  {
    return textures_;
  }


  TEX_T* ptr_d_;

protected:
  std::vector<TextureParams<DATA_T>> textures_;

  // helper, on CPU points to vector, on GPU points to device copy
  TextureParams<DATA_T>* textures_d_ = nullptr;

  // device pointer to the parameters malloced memory
  TextureParams<DATA_T>* params_d_;
};

#if __CUDACC__
#include "texture_helper.cu"
#endif


#endif //MPPIGENERIC_TEXTURE_HELPER_CUH
