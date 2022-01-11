#include "texture_helper.cuh"

template <class TEX_T, class DATA_T>
TextureHelper<TEX_T, DATA_T>::TextureHelper(int number, cudaStream_t stream) : Managed(stream)
{
  textures_.resize(number);
  textures_d_ = textures_.data();
}

template <class TEX_T, class DATA_T>
TextureHelper<TEX_T, DATA_T>::~TextureHelper()
{
  freeCudaMem();
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::GPUSetup()
{
  TEX_T* derived = static_cast<TEX_T*>(this);
  if (!GPUMemStatus_)
  {
    ptr_d_ = Managed::GPUSetup<TEX_T>(derived);
    // allocates memory to access params on the GPU by pointer
    HANDLE_ERROR(cudaMalloc(&params_d_, sizeof(TextureParams<DATA_T>) * textures_.size()));
    HANDLE_ERROR(cudaMemcpyAsync(&(ptr_d_->textures_d_), &(params_d_), sizeof(TextureParams<DATA_T>*),
                                 cudaMemcpyHostToDevice, this->stream_));
    cudaStreamSynchronize(this->stream_);
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;  // TODO should this be an exception?
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    for (int index = 0; index < textures_.size(); index++)
    {
      freeCudaMem(textures_[index]);
    }
    cudaFree(params_d_);
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::freeCudaMem(TextureParams<DATA_T>& texture)
{
  if (texture.allocated)
  {
    HANDLE_ERROR(cudaFreeArray(texture.array_d));
    HANDLE_ERROR(cudaDestroyTextureObject(texture.tex_d));
    texture.allocated = false;
    texture.use = false;
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::allocateCudaTexture(int index)
{
  // if already allocated deallocate
  if (this->GPUMemStatus_ && textures_[index].allocated)
  {
    freeCudaMem(textures_[index]);
  }
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::worldPoseToMapPose(const int index, const float3& input,
                                                                          float3& output)
{
  float3 diff = make_float3(input.x - textures_d_[index].origin.x, input.y - textures_d_[index].origin.y,
                            input.z - textures_d_[index].origin.z);
  float3* rotation_mat_ptr = textures_d_[index].rotations;
  output.x = (rotation_mat_ptr[0].x * diff.x + rotation_mat_ptr[0].y * diff.y + rotation_mat_ptr[0].z * diff.z);
  output.y = (rotation_mat_ptr[1].x * diff.x + rotation_mat_ptr[1].y * diff.y + rotation_mat_ptr[1].z * diff.z);
  output.z = (rotation_mat_ptr[2].x * diff.x + rotation_mat_ptr[2].y * diff.y + rotation_mat_ptr[2].z * diff.z);
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::mapPoseToTexCoord(const int index, const float3& input,
                                                                         float3& output)
{
  // TODO why 0.5??
  // from map frame to pixels [m] -> [px]
  output.x = input.x / textures_d_[index].resolution.x;
  output.y = input.y / textures_d_[index].resolution.y;
  output.z = input.z / textures_d_[index].resolution.z;

  // normalize pixel values
  output.x /= textures_d_[index].extent.width;
  output.y /= textures_d_[index].extent.height;
  if (textures_d_[index].extent.depth != 0)
  {
    output.z /= textures_d_[index].extent.depth;
  }
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::worldPoseToTexCoord(const int index, const float3& input,
                                                                           float3& output)
{
  float3 map;
  worldPoseToMapPose(index, input, map);
  mapPoseToTexCoord(index, map, output);
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::copyToDevice(bool synchronize)
{
  if (!this->GPUMemStatus_)
  {
    return;
  }

  // goes through and checks what needs to be copied and does it
  TEX_T* derived = static_cast<TEX_T*>(this);
  for (int i = 0; i < textures_.size(); i++)
  {
    TextureParams<DATA_T>* param = &textures_[i];

    // do the allocation and texture creation
    if (param->update_mem)
    {
      derived->allocateCudaTexture(i);
      derived->createCudaTexture(i, false);
    }
    // if allocated
    if (param->allocated)
    {
      // if we have new parameter values copy it over
      if (param->update_params)
      {
        derived->copyParamsToGPU(i, false);
      }

      // if we have updated data copy it over
      if (param->update_data)
      {
        // copies data to the GPU
        derived->copyDataToGPU(i, false);
      }
    }
  }
  if (synchronize)
  {
    cudaStreamSynchronize(this->stream_);
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::createCudaTexture(int index, bool sync)
{
  TextureParams<DATA_T>* cpu_param = &textures_[index];
  cpu_param->resDesc.res.array.array = cpu_param->array_d;

  HANDLE_ERROR(cudaCreateTextureObject(&(cpu_param->tex_d), &cpu_param->resDesc, &cpu_param->texDesc, NULL));

  cpu_param->allocated = true;
  cpu_param->update_mem = false;

  copyParamsToGPU(index, sync);
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::addNewTexture(const cudaExtent& extent)
{
  textures_.resize(textures_.size() + 1);
  textures_.back().extent = extent;

  if (this->GPUMemStatus_)
  {
    TEX_T* derived = static_cast<TEX_T*>(this);
    int index = textures_.size() - 1;

    // TODO resize the device side array

    derived->allocateCudaTexture(index);
    derived->createCudaTexture(index);
  }
}

template <class TEX_T, class DATA_T>
__device__ DATA_T TextureHelper<TEX_T, DATA_T>::queryTextureAtWorldPose(const int index, const float3& input)
{
  float3 tex_coords;
  worldPoseToTexCoord(index, input, tex_coords);
  TEX_T* derived = static_cast<TEX_T*>(this);
  return derived->queryTexture(index, tex_coords);
}

template <class TEX_T, class DATA_T>
__device__ DATA_T TextureHelper<TEX_T, DATA_T>::queryTextureAtMapPose(const int index, const float3& input)
{
  float3 tex_coords;
  mapPoseToTexCoord(index, input, tex_coords);
  TEX_T* derived = static_cast<TEX_T*>(this);
  return derived->queryTexture(index, tex_coords);
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateOrigin(int index, float3 new_origin)
{
  this->textures_[index].origin = new_origin;
  this->textures_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateRotation(int index, std::array<float3, 3>& new_rotation)
{
  this->textures_[index].rotations[0] = new_rotation[0];
  this->textures_[index].rotations[1] = new_rotation[1];
  this->textures_[index].rotations[2] = new_rotation[2];
  this->textures_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateResolution(int index, float resolution)
{
  this->textures_[index].resolution.x = resolution;
  this->textures_[index].resolution.y = resolution;
  this->textures_[index].resolution.z = resolution;
  this->textures_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateResolution(int index, float3 resolution)
{
  this->textures_[index].resolution.x = resolution.x;
  this->textures_[index].resolution.y = resolution.y;
  this->textures_[index].resolution.z = resolution.z;
  this->textures_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
bool TextureHelper<TEX_T, DATA_T>::setExtent(int index, cudaExtent& extent)
{
  // checks if the extent has changed and reallocates if yes
  TextureParams<DATA_T>* param = &textures_[index];
  if (param->extent.width != extent.width || param->extent.height != extent.height ||
      param->extent.depth != extent.depth)
  {
    // flag to update mem next time we should
    param->update_mem = true;
    this->textures_[index].extent = extent;
    return true;
  }
  return false;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::copyParamsToGPU(int index, bool sync)
{
  TextureParams<DATA_T>* cpu_param = &textures_[index];

  // Copy entire param structure over from CPU to GPU
  HANDLE_ERROR(cudaMemcpyAsync(&(params_d_[index]), cpu_param, sizeof(TextureParams<DATA_T>), cudaMemcpyHostToDevice,
                               this->stream_));
  cpu_param->update_params = false;
  if (sync)
  {
    cudaStreamSynchronize(this->stream_);
  }
}
