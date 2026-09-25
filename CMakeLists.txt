cmake_minimum_required(VERSION 3.18)
project(CUDA_Image_Pipeline CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED) # Optional, or use standard I/O if preferred

add_executable(cuda_image_pipeline src/main.cu)

target_link_libraries(cuda_image_pipeline ${CUDA_LIBRARIES} ${OpenCV_LIBS})
