#pragma once
// CUDA->HIP compat shim for the MPPI-Generic ROCm port.
//
// On ROCm/HIP this maps the CUDA runtime / cuRAND / cuFFT spellings the library
// uses onto their HIP equivalents and pulls in the HIP runtime. On NVIDIA it is
// a thin include of the real CUDA toolkit headers, so the CUDA build is byte
// for byte unchanged. This is the ONLY file that knows about HIP; everywhere
// else the sources keep the plain cuda*/curand*/cufft* spelling.
//
// The angle-bracket toolkit includes (<cuda_runtime.h>, <curand.h>, <cufft.h>,
// <device_launch_parameters.h>, <cooperative_groups.h>) are satisfied on HIP by
// the sibling shim headers in this directory; CMake puts this directory on the
// HIP include path ahead of the system includes so those names resolve here.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// libc host decls must win over HIP's __device__ memcpy/memset overloads once
// <hip/hip_runtime.h> is in scope (gpuRIR lesson), so include them first.
#include <cstring>
#include <cstdlib>

// The library and its tests call isnan/isinf/isfinite UNQUALIFIED inside
// templates (~70 sites: test_helper.h, base_plant.hpp, dynamics/cost models).
// nvcc/GCC resolve those to the global C-library names; clang's strict two-phase
// lookup does not find them at template definition. These using-declarations put
// the std overloads in the global namespace (host + device, verified) so the
// unqualified calls resolve, without editing every call site. Force-included on
// every HIP TU so the declarations precede all template definitions.
#include <cmath>
using std::isfinite;
using std::isinf;
using std::isnan;

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hipfft/hipfft.h>

// The library branches device-vs-host code on `#ifdef __CUDA_ARCH__` in ~60
// places (parallel index helpers, dynamics, math). HIP device compilation does
// NOT define __CUDA_ARCH__ (it uses __HIP_DEVICE_COMPILE__, which is 1 in the
// device pass and 0 in the host pass), so without this those device branches
// would silently collapse to the CPU fallback. Define __CUDA_ARCH__ only in the
// HIP device pass so the existing guards mean "device code" on both backends.
// This header is force-included on every HIP TU (CMAKE_HIP_FLAGS -include) so
// the define precedes any __CUDA_ARCH__ use.
// The library is header-only via the nvcc idiom `#if __CUDACC__ #include
// "foo.cu"` at the bottom of each foo.cuh (59 sites) -- the .cu definitions are
// pulled in only when the device compiler is active. hipcc defines __HIPCC__
// (both passes) but not __CUDACC__, so without this every device kernel/helper
// would be undeclared. __CUDACC__ means "compiled by the CUDA device compiler",
// the exact role hipcc plays here.
#if defined(__HIPCC__) && !defined(__CUDACC__)
#define __CUDACC__ 1
#endif

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 1
#endif

// Eigen has a native HIP device path (EIGEN_HIP_DEVICE_COMPILE) and a separate
// CUDA path that, seeing our __CUDA_ARCH__ above, would pull in CUDA's
// <math_constants.h> (absent on ROCm). EIGEN_NO_CUDA makes Eigen ignore
// __CUDA_ARCH__ and use its HIP path instead (it ships its own HIP
// math_constants.h). Must precede any Eigen include, hence the force-include.
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

// ---- runtime: error + device ----
#define cudaError_t hipError_t
#define cudaError hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
// hipFuncSetAttribute takes the kernel as `const void*` (CUDA accepts the typed
// kernel pointer). The call sites cast their templated __global__ pointer with
// MPPI_GPU_FUNC_PTR below (a function-like macro can't take the comma-bearing
// template arg).
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// ---- memory ----
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// ---- streams ----
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize

// ---- events ----
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// ---- 3D / pitched memory + arrays ----
#define cudaExtent hipExtent
#define cudaPos hipPos
#define cudaPitchedPtr hipPitchedPtr
#define make_cudaExtent make_hipExtent
#define make_cudaPos make_hipPos
#define make_cudaPitchedPtr make_hipPitchedPtr
#define cudaArray hipArray
#define cudaArray_t hipArray_t
#define cudaMallocArray hipMallocArray
#define cudaMalloc3DArray hipMalloc3DArray
#define cudaFreeArray hipFreeArray
#define cudaMemcpyToArray hipMemcpyToArray
#define cudaMemcpy2DToArrayAsync hipMemcpy2DToArrayAsync
#define cudaMemcpy3DAsync hipMemcpy3DAsync
#define cudaMemcpy3DParms hipMemcpy3DParms

// ---- textures ----
#define cudaTextureObject_t hipTextureObject_t
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject
#define cudaResourceDesc hipResourceDesc
#define cudaTextureDesc hipTextureDesc
#define cudaResourceTypeArray hipResourceTypeArray
#define cudaChannelFormatDesc hipChannelFormatDesc
#define cudaChannelFormatKindFloat hipChannelFormatKindFloat
#define cudaCreateChannelDesc hipCreateChannelDesc
#define cudaTextureAddressMode hipTextureAddressMode
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaAddressModeBorder hipAddressModeBorder
#define cudaAddressModeWrap hipAddressModeWrap
#define cudaFilterModeLinear hipFilterModeLinear
#define cudaFilterModePoint hipFilterModePoint
#define cudaReadModeElementType hipReadModeElementType

// ---- cuRAND (host generator API) ----
#define curandGenerator_t hiprandGenerator_t
#define curandStatus_t hiprandStatus_t
#define curandCreateGenerator hiprandCreateGenerator
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandSetGeneratorOffset hiprandSetGeneratorOffset
#define curandSetStream hiprandSetStream
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateLogNormal hiprandGenerateLogNormal
#define curandGeneratePoisson hiprandGeneratePoisson
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
// cuRAND status enum -> hipRAND. hipRAND mirrors the cuRAND status codes.
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_STATUS_VERSION_MISMATCH HIPRAND_STATUS_VERSION_MISMATCH
#define CURAND_STATUS_NOT_INITIALIZED HIPRAND_STATUS_NOT_INITIALIZED
#define CURAND_STATUS_ALLOCATION_FAILED HIPRAND_STATUS_ALLOCATION_FAILED
#define CURAND_STATUS_TYPE_ERROR HIPRAND_STATUS_TYPE_ERROR
#define CURAND_STATUS_OUT_OF_RANGE HIPRAND_STATUS_OUT_OF_RANGE
#define CURAND_STATUS_LENGTH_NOT_MULTIPLE HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
#define CURAND_STATUS_DOUBLE_PRECISION_REQUIRED HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define CURAND_STATUS_LAUNCH_FAILURE HIPRAND_STATUS_LAUNCH_FAILURE
#define CURAND_STATUS_PREEXISTING_FAILURE HIPRAND_STATUS_PREEXISTING_FAILURE
#define CURAND_STATUS_INITIALIZATION_FAILED HIPRAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_ARCH_MISMATCH HIPRAND_STATUS_ARCH_MISMATCH
#define CURAND_STATUS_INTERNAL_ERROR HIPRAND_STATUS_INTERNAL_ERROR

// ---- cuFFT ----
#include <hip/hip_complex.h>
#define cuComplex hipComplex
#define cuFloatComplex hipFloatComplex
#define cufftHandle hipfftHandle
#define cufftResult hipfftResult
#define cufftComplex hipfftComplex
#define cufftReal hipfftReal
#define cufftPlan1d hipfftPlan1d
#define cufftExecC2R hipfftExecC2R
#define cufftSetStream hipfftSetStream
#define cufftDestroy hipfftDestroy
#define CUFFT_C2R HIPFFT_C2R
#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_C2C HIPFFT_C2C
// cuFFT result enum -> hipFFT.
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_INVALID_VALUE HIPFFT_INVALID_VALUE
#define CUFFT_INTERNAL_ERROR HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED HIPFFT_SETUP_FAILED
#define CUFFT_INVALID_SIZE HIPFFT_INVALID_SIZE
#define CUFFT_UNALIGNED_DATA HIPFFT_UNALIGNED_DATA
#define CUFFT_INCOMPLETE_PARAMETER_LIST HIPFFT_INCOMPLETE_PARAMETER_LIST
#define CUFFT_INVALID_DEVICE HIPFFT_INVALID_DEVICE
#define CUFFT_PARSE_ERROR HIPFFT_PARSE_ERROR
#define CUFFT_NO_WORKSPACE HIPFFT_NO_WORKSPACE
#define CUFFT_NOT_IMPLEMENTED HIPFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED HIPFFT_NOT_SUPPORTED
// CUFFT_LICENSE_ERROR, CUFFT_MISSING_DEPENDENCY, CUFFT_NV{RTC,JITLINK,SHMEM}_FAILURE
// have no hipFFT equivalent; their switch cases in gpu_err_chk.cuh are USE_HIP-guarded.

// Cast a __global__ kernel pointer for hipFuncSetAttribute (needs const void*).
// Variadic so a comma-bearing template-id (kernel<A, B, C>) is one argument.
#define MPPI_GPU_FUNC_PTR(...) ((const void*)(__VA_ARGS__))

#else  // CUDA build: unchanged

#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>

// CUDA's cudaFuncSetAttribute accepts the typed kernel pointer directly.
#define MPPI_GPU_FUNC_PTR(...) (__VA_ARGS__)

#endif
