#ifndef CUDA_TESTING_GPU_ERR_CHK_CUH
#define CUDA_TESTING_GPU_ERR_CHK_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For block idx and thread idx, etc
#include <iostream>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline void __cudaCheckError( const char *file, const int line )
{
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }



#endif //! CUDA_TESTING_GPU_ERR_CHK_CUH