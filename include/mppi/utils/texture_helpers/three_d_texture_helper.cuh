//
// Created by jason on 1/10/22.
//

#ifndef MPPIGENERIC_THREE_D_TEXTURE_HELPER_CUH
#define MPPIGENERIC_THREE_D_TEXTURE_HELPER_CUH

#include <mppi/utils/texture_helpers/texture_helper.cuh>

// TODO needs to 3D wrap around angles, i.e set 3D to wrap

template <class DATA_T>
class ThreeDTextureHelper : public TextureHelper<ThreeDTextureHelper<DATA_T>, DATA_T>
{
public:
  ThreeDTextureHelper(int number, cudaStream_t stream = 0);

  void allocateCudaTexture(int index) override;

  void updateTexture(const int index, const int z_index, std::vector<DATA_T>& data);
  void updateTexture(const int index, const int z_index,
                     const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                                      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
                         values);
  bool setExtent(int index, cudaExtent& extent) override;
  void copyDataToGPU(int index, bool sync = false) override;

  std::vector<std::vector<DATA_T>> getCpuValues()
  {
    return cpu_values_;
  }
  std::vector<std::vector<bool>> getLayerCopy()
  {
    return layer_copy_;
  }

  __device__ DATA_T queryTexture(const int index, const float3& point);

protected:
  // layer -> 3D array -> actual texture lookup thing
  std::vector<std::vector<DATA_T>> cpu_values_;
  std::vector<std::vector<bool>> layer_copy_;  // indicator what 2D part of the 3D array needs to be copied over
};

#if __CUDACC__
#include "three_d_texture_helper.cu"
#endif

#endif  // MPPIGENERIC_THREE_D_TEXTURE_HELPER_CUH
