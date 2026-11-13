# Needed for methods to be built into libraries in Release mode under nvcc. The
# flag is a GCC host-codegen hint that clang (hence hipcc) does not accept, and
# it is not needed for the HIP build, so apply it on CUDA only.
if (NOT USE_HIP)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-guess-branch-probability")
endif()

add_subdirectory(cartpole)
add_subdirectory(autorally)
add_subdirectory(double_integrator)
add_subdirectory(quadrotor)