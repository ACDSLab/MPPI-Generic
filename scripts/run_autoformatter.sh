#! /usr/bin/env bash

# Format cuda header files
echo "Formatting cuh files..."
find . -name "*.cuh" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i --assume-filename=cu %"

# Format cuda source files
echo "Formatting cu files..."
find . -name "*.cu" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"

# Format cpp source files
echo "Formatting cpp files..."
find . -name "*.cpp" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"

# Format h source files
echo "Formatting h files..."
find . -name "*.h" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"

# Format hpp source files
echo "Formatting hpp files..."
find . -name "*.hpp" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"
