//
// Created by jason on 4/25/22.
//

#ifndef MPPIGENERIC_CUDA_MATH_UTILS_CUH
#define MPPIGENERIC_CUDA_MATH_UTILS_CUH

__host__ __device__ inline float2 operator*(const float2& a, const float& b)
{
  return make_float2(a.x * b, a.y * b);
}
__host__ __device__ inline float3 operator*(const float3& a, const float& b)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline float4 operator*(const float4& a, const float& b)
{
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ inline float2 operator+(const float2& a, const float& b)
{
  return make_float2(a.x + b, a.y + b);
}
__host__ __device__ inline float3 operator+(const float3& a, const float& b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
__host__ __device__ inline float4 operator+(const float4& a, const float& b)
{
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__host__ __device__ inline float2 operator-(const float2& a, const float& b)
{
  return make_float2(a.x - b, a.y - b);
}
__host__ __device__ inline float3 operator-(const float3& a, const float& b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
__host__ __device__ inline float4 operator-(const float4& a, const float& b)
{
  return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

__host__ __device__ inline float2 operator+(const float2& a, const float2& b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float4 operator+(const float4& a, const float4& b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline float2 operator-(const float2& a, const float2& b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float4 operator-(const float4& a, const float4& b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ inline float2 operator*(const float2& a, const float2& b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float4 operator*(const float4& a, const float4& b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ inline float2& operator*=(float2& a, const float& b)
{
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline float3& operator*=(float3& a, const float& b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  return a;
}
__host__ __device__ inline float4& operator*=(float4& a, const float& b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}

__host__ __device__ inline float2& operator+=(float2& a, const float2& b)
{
  a.x += b.x;
  a.y += b.y;
  return a;
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__host__ __device__ inline float4& operator+=(float4& a, const float4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

__host__ __device__ inline float dot(const float2& a, const float2& b)
{
  return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline float cross(const float2& a, const float2& b)
{
  return a.x * b.y - a.y * b.x;
}

__host__ __device__ inline float norm(const float2& a)
{
  return sqrtf(dot(a, a));
}

__host__ __device__ inline float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float norm(const float3& a)
{
  return sqrtf(dot(a, a));
}

__host__ __device__ inline float dot(const float4& a, const float4& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ inline float norm(const float4& a)
{
  return sqrtf(dot(a, a));
}

__host__ __device__ inline float2 operator-(const float2& a)
{
  return make_float2(-a.x, -a.y);
}

__host__ __device__ inline float3 operator-(const float3& a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float4 operator-(const float4& a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

__host__ __device__ inline bool operator==(const float2& lhs, const float2& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

__host__ __device__ inline bool operator==(const float3& lhs, const float3& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

__host__ __device__ inline bool operator==(const float4& lhs, const float4& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.w == rhs.w);
}
#endif  // MPPIGENERIC_CUDA_MATH_UTILS_CUH
