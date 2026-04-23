#! /usr/bin/env bash

# if no arguments passed in, use current folder as path
if [ "$1" == "" ]; then
  paths=("./")
else
  paths=( "$@" )
fi

for dir in "${paths[@]}"; do
  # Format cuda header files
  echo "Formatting cuh files..."
  find $dir -name "*.cuh" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i --assume-filename=cu %"

  # Format cuda source files
  echo "Formatting cu files..."
  find $dir -name "*.cu" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"

  # Format cpp source files
  echo "Formatting cpp files..."
  find $dir -name "*.cpp" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"

  # Format h source files
  echo "Formatting h files..."
  find $dir -name "*.h" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"

  # Format hpp source files
  echo "Formatting hpp files..."
  find $dir -name "*.hpp" ! -path "*build*" ! -path "*submodules*" | xargs -I % bash -c "clang-format --style=file -i %"
  shift # remove argument 1 and shift all arguments down
done
