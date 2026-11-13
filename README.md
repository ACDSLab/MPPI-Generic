# MPPI-Generic
MPPI-Generic is a C++/CUDA header-only library implementation of Model Predictive Path Integral Control (MPPI) by [Williams et al.](https://ieeexplore.ieee.org/document/8558663)

## Citation
If you use this library for research purposes, please cite the following [paper](https://arxiv.org/abs/2409.07563):
```
@misc{vlahov2024mppi,
      title={MPPI-Generic: A CUDA Library for Stochastic Trajectory Optimization},
      author={Bogdan Vlahov and Jason Gibson and Manan Gandhi and Evangelos A. Theodorou},
      year={2024},
      eprint={2409.07563},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2409.07563},
}
```

## Requirements
MPPI-Generic relies on the following:
* An NVIDIA GPU
* GCC/G++
* CUDA 10 or newer (CUDA 11.7+ is recommended but our library is compatible back to CUDA 10)
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [CMake](https://cmake.org/) 3.10 or newer
* git and git-lfs
### Unit tests requirements
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)
* python-pil

## Prerequisite Setup (Ubuntu)
1. Follow the instructions to install CUDA provided [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

2. Install all the other prerequisites through `apt-get`:
```bash
sudo apt-get install libeigen3-dev git git-lfs cmake gcc
# Setup git lfs if it is the first you have installed it
git lfs install
# extra installs if you are wanting to build unit tests
sudo apt-get install libyaml-cpp-dev python3-pil
```

Note: If using Pop!\_OS you can `sudo apt install system76-cuda` instead of installing CUDA manually.

## Download the repo
```bash
cd /path/to/repos
git clone https://github.com/ACDSLab/MPPI-Generic.git
cd MPPI-Generic
git submodule update --init --recursive
```
## Building MPPI-Generic with tests

The default is to build the library with tests OFF.
If you would like to turn on the tests when building, pass the flag `-DMPPI_BUILD_TESTS=ON` when configuring cmake.

```bash
mkdir build
cd build
cmake -DMPPI_BUILD_TESTS=ON ..
make
make test
```

## Building MPPI-Generic for AMD GPUs (ROCm)

The GPU sources can also be built with HIP for AMD GPUs. Install [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) (it provides hipRAND, hipFFT and the HIP compiler), then configure with `-DUSE_HIP=ON` and the target architecture of your GPU:

```bash
mkdir build
cd build
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DMPPI_BUILD_TESTS=ON ..
make
make test
```

If `CMAKE_HIP_ARCHITECTURES` is omitted, CMake detects the architecture of the installed GPU. The CUDA build is unchanged when `USE_HIP` is off.

## Acknowledgements
Approved for Public Release, Distribution Unlimited.
