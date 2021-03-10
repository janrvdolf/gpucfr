cmake_minimum_required(VERSION 3.16)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

project(gpucfr CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB SOURCE_FILES
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        )

add_executable(gpucfr main.cu ${SOURCE_FILES})
set_target_properties(gpucfr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpucfr_correctness main_correctness.cu ${SOURCE_FILES})
set_target_properties(gpucfr_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)