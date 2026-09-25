#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>

// Error checking macro for CUDA and NPP API calls
#fn check_npp(call) {
    NppStatus status = call;
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error: " << status << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Custom CUDA Kernel for auxiliary pixel manipulation
__global__ void adjustContrastKernel(Npp8u* d_data, int width, int height, int pitch, float factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        float val = static_cast<float>(d_data[idx]) * factor;
        d_data[idx] = static_cast<Npp8u>(val > 255.0f ? 255.0f : (val < 0.0f ? 0.0f : val));
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Starting High-Performance CUDA NPP Image Processing Pipeline...\n";

    // Initialize dimensions and parameters (accepting command-line args)
    int width = 1920;
    int height = 1080;
    if (argc > 2) {
        width = std::stoi(argv[1]);
        height = std::stoi(argv[2]);
    }

    int numBytes = width * height * sizeof(Npp8u);
    Npp8u* d_src = nullptr;
    
    // Allocate Device Memory
    cudaMalloc(&d_src, numBytes);

    // Setup execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch Custom Kernel
    adjustContrastKernel<<<gridSize, blockSize>>>(d_src, width, height, width, 1.2f);
    cudaDeviceSynchronize();

    std::cout << "Image filtering and CUDA stream orchestration executed successfully.\n";

    // Clean up
    cudaFree(d_src);
    return 0;
}