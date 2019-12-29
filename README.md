# MPPI-Generic
Generic implementation of Model Predictive Path Integral Control

## Requirements
* GCC/G++ >= 6.5
* CMake >= 3.8
* CUDA Toolkit == 10.0

## Third Party Libs:
* cnpy
* yaml
* eigen3
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