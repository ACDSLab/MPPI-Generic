set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add debug flags so cuda-gdb can be used to stop inside a kernel.
# NOTE: You may have to run make multiple times for it to compile successfully.
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G --keep")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
# required for curand on some systems
find_package(CUDA REQUIRED)

if(NOT ${CUDA_curand_LIBRARY} MATCHES "NOTFOUND")
    set(CURAND_LIBRARY ${CUDA_curand_LIBRARY})
endif()

# Generate name for MPPI header library
set(MPPI_HEADER_LIBRARY_NAME mppi_header_only_lib)

# Generate variable for all the extra cuda libraries we use
set(MPPI_GENERIC_CUDA_EXTRA_LIBS "")
list(APPEND MPPI_GENERIC_CUDA_EXTRA_LIBS ${CUDA_curand_LIBRARY} ${CUDA_CUFFT_LIBRARIES})

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Autodetect Cuda Architecture on system and add to executables
# More info for autodetection:
# https://stackoverflow.com/questions/35485087/determining-which-gencode-compute-arch-values-i-need-for-nvcc-within-cmak
if (NOT DEFINED MPPI_ARCH_FLAGS)
  CUDA_SELECT_NVCC_ARCH_FLAGS(MPPI_ARCH_FLAGS ${CUDA_ARCH_LIST})

  if (MPPI_ARCH_FLAGS STREQUAL "")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=32 -arch=sm_35 -w")
  else()
    string(REGEX REPLACE "-gencode;arch" "-gencode=arch" MPPI_ARCH_FLAGS "${MPPI_ARCH_FLAGS}")
    string(REPLACE ";" " " MPPI_ARCH_FLAGS "${MPPI_ARCH_FLAGS}")
    # string(REGEX REPLACE "^-gencode=arch" "-arch" MPPI_ARCH_FLAGS "${MPPI_ARCH_FLAGS}")
    message(STATUS "Additional Architectures: ${MPPI_ARCH_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MPPI_ARCH_FLAGS}")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_52 -w")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -code=sm_72 -arch=compute_52 -w")
  endif()
else()
  message(STATUS "Auotdetection already ran and found ${MPPI_ARCH_FLAGS}.")
endif()
