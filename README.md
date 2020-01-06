# MPPI-Generic
Generic implementation of Model Predictive Path Integral Control

## Requirements
* GCC/G++ >= 4.5
* CMake >= 3.8
* CUDA Toolkit == 10.0
* Eigen >= 3.3.4

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

## Building MPPI-Generic with tests

Pass the flag `DBUILD_TESTS=ON` when configuring cmake.

```bash
cmake .. -DBUILD_TESTS=ON
```

## TODO

### Timing and Logging

We can utilize the cuda event API to time kernel executions. CPU time can be
measured using the ```chrono``` system import.