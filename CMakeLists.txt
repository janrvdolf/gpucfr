cmake_minimum_required(VERSION 3.17)
project(gpucfr CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR})

set (SOURCE_FILES
        cuda_base.cpp
        information_set.cu
        efg_node.cu
        gpucfr.cu
        )

add_executable(gpucfr main.cu ${SOURCE_FILES})
set_target_properties(
        gpucfr
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)