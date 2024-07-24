# MPPI-Generic
Generic implementation of Model Predictive Path Integral Control

## Requirements
MPPI-Generic relies on the following:
* An NVIDIA GPU
* GCC/G++
* CUDA 10 or newer [(Installation instructions)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
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
git clone https://github.gatech.edu/ACDS/MPPI-Generic.git
cd MPPI-Generic
git submodule update --init --recursive
```
## Building MPPI-Generic with tests

The default is to build the library with tests OFF.
If you would like to turn on the tests when building, pass the flag `-DBUILD_TESTS=ON` when configuring cmake.

```bash
mkdir build
cd build
cmake .. -DBUILD_TESTS=ON
make
make test
```
