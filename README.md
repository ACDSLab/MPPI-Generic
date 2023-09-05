# MPPI-Generic
Generic implementation of Model Predictive Path Integral Control

## Requirements
* GCC/G++ >= 4.5
* CMake >= 3.8
* CUDA Toolkit == 10.0
* Eigen >= 3.3.4
* python-pil

## Install instructions (Ubuntu 16.04)

The default cmake and Eigen in Ubuntu are not versions required by this library on 16.04.

### CMake
* Download a version of cmake after 3.8. Generally the latest version is recommended
* If you don't want to delete system CMake, follow these instructions:
```bash
cd cmake-3.15.0-rc3 # Go into downloaded cmake folder
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local # Installs cmake to the ~/.local directory rather than /usr/local/ where CMake is installed by default
make install # Build and install cmake to ~/.local
```
* You might need to reopen your terminal for it to take full effect
* Check to see CMake is up to date
```bash
cmake --version
```

### Eigen
* Download the latest version of Eigen [here](http://eigen.tuxfamily.org/index.php)
* If you don't want to delete system Eigen, follow these instructions:
```bash
cd eigen-3.3.7 # Go into downloaded eigen folder
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local # Installs Eigen to the ~/.local directory rather than /usr/local/ where Eigen is installed by default
make install # Install Eigen to ~/.local
```
## Third Party Libs:
* cnpy
* yaml
* ros ( wrappers only)

## Install instructions (Ubuntu 20.04)

### Install Dependencies

```bash
sudo apt install cmake libeigen3-dev python3-pil
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

The default is to build the library with tests ON. If you would like to turn off the tests when building, pass the flag `DBUILD_TESTS=OFF` when configuring cmake.

```bash
mkdir build
cd build
cmake .. -DBUILD_TESTS=OFF
make
```

## TODO

### Timing and Logging

We can utilize the cuda event API to time kernel executions. CPU time can be
measured using the ```chrono``` system import.
