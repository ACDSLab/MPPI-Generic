//
// Created by jason on 1/5/22.
//

#include "two_d_texture_helper.cuh"

template <class DATA_T>
TwoDTextureHelper<DATA_T>::TwoDTextureHelper(int number, cudaStream_t stream)
  : TextureHelper<TwoDTextureHelper<DATA_T>, DATA_T>(number, stream)
{
  cpu_values_.resize(number);
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::allocateCudaTexture(int index)
{
  TextureHelper<TwoDTextureHelper<DATA_T>, DATA_T>::allocateCudaTexture(index);

  TextureParams<DATA_T>* param = &this->textures_[index];

  int w = param->extent.width;
  int h = param->extent.height;

  HANDLE_ERROR(cudaMallocArray(&(param->array_d), &(param->channelDesc), w, h));
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(const int index, std::vector<DATA_T>& values)
{
  TextureParams<DATA_T>* param = &this->textures_[index];
  int w = param->extent.width;
  int h = param->extent.height;

  // check that the sizes are correct
  if (values.size() != w * h)
  {
    throw std::runtime_error(std::string("Error: invalid size to updateTexture ") + std::to_string(values.size()) +
                             " != " + std::to_string(w * h));
  }

  // copy over values to cpu side holder
  // TODO handle row/column major data
  cpu_values_[index].resize(w * h);
  if (param->column_major)
  {
    for (int i = 0; i < h; i++)
    {
      for (int j = 0; j < w; j++)
      {
        int columnMajorIndex = i * w + j;
        int rowMajorIndex = j * h + i;
        cpu_values_[index][rowMajorIndex] = values[columnMajorIndex];
      }
    }
  }
  else
  {
    // std::copy(values.begin(), values.end(), cpu_values_[index].begin());
    cpu_values_[index] = std::move(values);
  }
  // tells the object to copy it over next time that happens
  param->update_data = true;
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(const int index, std::vector<DATA_T>& data, cudaExtent& extent)
{
  setExtent(index, extent);
  updateTexture(index, data);
}

template <class DATA_T>
__device__ DATA_T TwoDTextureHelper<DATA_T>::queryTexture(const int index, const float3& point)
{
  return tex2D<DATA_T>(this->textures_d_[index].tex_d, point.x, point.y);
}

template <class DATA_T>
bool TwoDTextureHelper<DATA_T>::setExtent(int index, cudaExtent& extent)
{
  if (extent.depth != 0)
  {
    throw std::runtime_error(std::string("Error: extent in setExtent invalid,"
                                         " cannot use depth != 0 in 2D texture: using ") +
                             std::to_string(extent.depth));
  }

  return TextureHelper<TwoDTextureHelper<DATA_T>, DATA_T>::setExtent(index, extent);
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::copyDataToGPU(int index, bool sync)
{
  TextureParams<DATA_T>* param = &this->textures_[index];
  int w = param->extent.width;
  int h = param->extent.height;
  HANDLE_ERROR(cudaMemcpy2DToArrayAsync(param->array_d, 0, 0, cpu_values_[index].data(), w * sizeof(DATA_T),
                                        w * sizeof(DATA_T), h, cudaMemcpyHostToDevice, this->stream_));
  if (sync)
  {
    cudaStreamSynchronize(this->stream_);
  }
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(
    const int index, const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                                      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
                         values)
{
  TextureParams<DATA_T>* param = &this->textures_[index];
  int w = param->extent.width;
  int h = param->extent.height;
  cpu_values_[index].resize(w * h);
  if (param->column_major)
  {
    for (int j = 0; j < w; j++)
    {
      for (int i = 0; i < h; i++)
      {
        int columnMajorIndex = j * h + i;
        int rowMajorIndex = i * w + j;
        cpu_values_[index][rowMajorIndex] = values.data()[columnMajorIndex];
      }
    }
  }
  else
  {
    memcpy(cpu_values_[index].data(), values.data(), values.size() * sizeof(DATA_T));
  }
  // tells the object to copy it over next time that happens
  param->update_data = true;
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(
    int index,
    const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                     Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        values,
    cudaExtent& extent)
{
  setExtent(index, extent);
  updateTexture(index, values);
}
