# High-Performance Parallel Image Processing Pipeline using CUDA NPP

This project implements an enterprise-grade parallel image processing and filtering pipeline leveraging NVIDIA Performance Primitives (NPP) and custom CUDA kernels.

## Features
- Accelerated image transformations and filtering operations.
- Optimized host-to-device memory allocation routines.
- Integration of custom CUDA kernels alongside NPP runtime APIs.
- Adheres strictly to the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

## Building and Execution

### Prerequisites
- NVIDIA CUDA Toolkit (v11.0+)
- CMake (v3.18+)
- Compatible NVIDIA GPU (Compute Capability 3.5+)

### Build Instructions
```bash
mkdir build
cd build
cmake ..
make
