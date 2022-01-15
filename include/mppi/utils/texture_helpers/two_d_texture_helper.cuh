//
// Created by jason on 1/5/22.
//

#ifndef MPPIGENERIC_TWO_D_TEXTURE_HELPER_CUH
#define MPPIGENERIC_TWO_D_TEXTURE_HELPER_CUH

#include <mppi/utils/texture_helpers/texture_helper.cuh>

template <class DATA_T>
class TwoDTextureHelper : public TextureHelper<TwoDTextureHelper<DATA_T>, DATA_T>
{
public:
  TwoDTextureHelper(int number, cudaStream_t stream = 0);

  void allocateCudaTexture(int index) override;

  void updateTexture(int index,
                     const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                                      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>,
                     cudaExtent& extent);
  void updateTexture(const int index, const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                                                       Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>);
  void updateTexture(const int index, std::vector<DATA_T>& data);
  void updateTexture(const int index, std::vector<DATA_T>& data, cudaExtent& extent);
  bool setExtent(int index, cudaExtent& extent) override;
  void copyDataToGPU(int index, bool sync = false);

  std::vector<std::vector<DATA_T>> getCpuValues()
  {
    return cpu_values_;
  }

  __device__ DATA_T queryTexture(const int index, const float3& point);

protected:
  std::vector<std::vector<DATA_T>> cpu_values_;
};

#if __CUDACC__
#include "two_d_texture_helper.cu"
#endif

#endif  // MPPIGENERIC_TWO_D_TEXTURE_HELPER_CUH
